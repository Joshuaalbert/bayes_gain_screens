import matplotlib

matplotlib.use('Agg')

import os
import sys
import numpy as np
import pylab as plt
import argparse
from timeit import default_timer
from jax import numpy as jnp, jit, random, vmap, tree_map
import logging
import astropy.units as au

from bayes_gain_screens.utils import poly_smooth, wrap, link_overwrite, windowed_mean, curv, \
    weighted_polyfit
from bayes_gain_screens.outlier_detection import detect_tec_outliers

logger = logging.getLogger(__name__)

from bayes_gain_screens.plotting import add_colorbar_to_axes, animate_datapack
from h5parm import DataPack
from h5parm.utils import make_soltab

from jaxns.prior_transforms import UniformPrior, PriorChain, HalfLaplacePrior, DeterministicTransformPrior, NormalPrior
from jaxns.nested_sampling import NestedSampler
from jaxns.utils import marginalise_dynamic, estimate_map, resample, marginalise_static, chunked_pmap

TEC_CONV = -8.4479745e6  # mTECU/Hz


def log_laplace(x, mean, uncert):
    dx = jnp.abs(x - mean) / uncert
    return -jnp.log(2. * uncert) - dx


def log_normal(x, mean, scale):
    dx = (x - mean) / scale
    return -0.5 * jnp.log(2. * jnp.pi) - jnp.log(scale) - 0.5 * dx * dx


def unconstrained_solve(freqs, key, phase_obs, phase_outliers):
    key1, key2, key3, key4 = random.split(key, 4)
    Nt, Nf = phase_obs.shape

    def log_likelihood(tec0, dtec, const, uncert, **kwargs):
        tec = jnp.concatenate([tec0[None], tec0 + jnp.cumsum(dtec)])
        t = freqs - jnp.min(freqs)
        t /= t[-1]
        uncert = uncert[0] + (uncert[1] - uncert[0]) * t
        phase = tec[:, None] * (TEC_CONV / freqs) + const
        logL = jnp.sum(jnp.where(phase_outliers, 0., log_normal(wrap(wrap(phase)-wrap(phase_obs)), 0., uncert)))
        return logL

    tec0 = UniformPrior('tec0', -300., 300.)
    # 30mTECU/30seconds is the maximum change
    dtec = UniformPrior('dtec', -30. * jnp.ones(Nt - 1), 30. * jnp.ones(Nt - 1))
    const = UniformPrior('const', -jnp.pi, jnp.pi)
    uncert = HalfLaplacePrior('uncert', 0.5*jnp.ones(2))
    prior_chain = PriorChain(tec0, dtec, const, uncert)

    ns = NestedSampler(log_likelihood,
                       prior_chain,
                       sampler_name='slice',
                       num_live_points=20 * prior_chain.U_ndims,
                       sampler_kwargs=dict(num_slices=prior_chain.U_ndims * 4))

    results = ns(key=key1, termination_evidence_frac=0.3)

    ESS = 900  # emperically estimated for this problem

    def marginalisation(tec0, dtec, const, uncert, **kwargs):
        tec = jnp.concatenate([tec0[None], tec0 + jnp.cumsum(dtec)])
        return tec, tec ** 2, jnp.cos(const), jnp.sin(const), jnp.mean(uncert)

    tec_mean, tec2_mean, const_real, const_imag, uncert_mean = marginalise_static(key2, results.samples, results.log_p,
                                                                                  ESS, marginalisation)

    tec_std = jnp.sqrt(tec2_mean - tec_mean ** 2)
    const_mean = jnp.arctan2(const_imag, const_real)

    def marginalisation(const, **kwargs):
        return wrap(wrap(const) - wrap(const_mean)) ** 2

    const_var = marginalise_static(key2, results.samples, results.log_p, ESS, marginalisation)
    const_std = jnp.sqrt(const_var)

    return tec_mean, tec_std, const_mean * jnp.ones(Nt), const_std * jnp.ones(Nt), uncert_mean * jnp.ones(Nt)


def constrained_solve(freqs, key, phase_obs, phase_outliers, const_mean, const_std):
    key1, key2, key3, key4 = random.split(key, 4)
    Nt, Nf = phase_obs.shape

    def log_likelihood(tec0, dtec, const, uncert, **kwargs):
        tec = jnp.concatenate([tec0[None], tec0 + jnp.cumsum(dtec)])
        t = freqs - jnp.min(freqs)
        t /= t[-1]
        uncert = uncert[0] + (uncert[1] - uncert[0])*t
        phase = tec[:, None] * (TEC_CONV / freqs) + const
        logL = jnp.sum(jnp.where(phase_outliers, 0., log_normal(wrap(wrap(phase) - wrap(phase_obs)), 0., uncert)))
        return logL

    tec0 = UniformPrior('tec0', -300., 300.)
    # 30mTECU/30seconds is the maximum change
    dtec = UniformPrior('dtec', -30. * jnp.ones(Nt - 1), 30. * jnp.ones(Nt - 1))
    const = NormalPrior('const', jnp.mean(const_mean), jnp.mean(const_std))
    uncert = HalfLaplacePrior('uncert', 0.5*jnp.ones(2))
    prior_chain = PriorChain(tec0, dtec, const, uncert)

    ns = NestedSampler(log_likelihood,
                       prior_chain,
                       sampler_name='slice',
                       num_live_points=20 * prior_chain.U_ndims,
                       sampler_kwargs=dict(num_slices=prior_chain.U_ndims * 4))

    results = ns(key=key1, termination_evidence_frac=0.3)

    ESS = 900  # emperically estimated for this problem

    def marginalisation(tec0, dtec, const, uncert, **kwargs):
        tec = jnp.concatenate([tec0[None], tec0 + jnp.cumsum(dtec)])
        return tec, tec ** 2, jnp.cos(const), jnp.sin(const), jnp.mean(uncert)

    tec_mean, tec2_mean, const_real, const_imag, uncert_mean = marginalise_static(key2, results.samples, results.log_p,
                                                                                  ESS, marginalisation)

    tec_std = jnp.sqrt(tec2_mean - tec_mean ** 2)
    const_mean = jnp.arctan2(const_imag, const_real)

    def marginalisation(const, **kwargs):
        return wrap(wrap(const) - wrap(const_mean)) ** 2

    const_var = marginalise_static(key2, results.samples, results.log_p, ESS, marginalisation)
    const_std = jnp.sqrt(const_var)

    return tec_mean, tec_std, const_mean * jnp.ones(Nt), const_std * jnp.ones(Nt)


def solve_and_smooth(gain_outliers, phase_obs, times, freqs):
    logger.info("Performing solve for tec and const from phases.")
    Nd, Na, Nf, Nt = phase_obs.shape

    logger.info("Number of nan: {}".format(jnp.sum(jnp.isnan(phase_obs))))
    logger.info("Number of inf: {}".format(jnp.sum(jnp.isinf(phase_obs))))

    # blocksize chosen to maximise Fisher information, which is 2 for tec+const, and 3 for tec+const+clock
    blocksize = 2

    remainder = Nt % blocksize
    if remainder != 0:
        if remainder < Nt:
            raise ValueError(f"Block size {blocksize} too big for number of timesteps {Nt}.")
        (gain_outliers, phase_obs) = tree_map(lambda x: jnp.concatenate([x, x[..., -remainder:]], axis=-1),
                                              (gain_outliers, phase_obs))
        Nt = Nt + remainder
        times = jnp.concatenate([times, times[-1] + jnp.arange(1, 1 + remainder) * jnp.mean(jnp.diff(times))])

    def _reshape(x):
        # Nd,Na,Nf,Nt//blocksize,blocksize
        x = jnp.reshape(x, (Nd, Na, Nf, Nt // blocksize, blocksize))
        # Nd, Na, Nt//blocksize, blocksize, Nf
        x = x.transpose((0, 1, 3, 4, 2))
        _shape = x.shape
        x = x.reshape((Nd * Na * (Nt // blocksize), blocksize, Nf))
        return x

    # [Nd*Na*(Nt//blocksize), blocksize, Nf]
    (gain_outliers, phase_obs) = tree_map(_reshape, (gain_outliers, phase_obs))

    T = Nd * Na * (Nt // blocksize)  # Nd * Na * (Nt // blocksize)
    keys = random.split(random.PRNGKey(int(1000 * default_timer())), T)

    # [Nd*Na*(Nt//blocksize), blocksize], [# Nd*Na*(Nt//blocksize), blocksize]
    tec_mean, tec_std, const_mean, const_std, uncert_mean = chunked_pmap(
        lambda *args: unconstrained_solve(freqs, *args),
        keys,
        phase_obs,
        gain_outliers)  # Nd*Na*(Nt//blocksize), blocksize

    const_weights = 1. / const_std ** 2

    def smooth(y, weights):
        y = y.reshape((Nd * Na, Nt))  # Nd*Na,Nt
        weights = weights.reshape((Nd * Na, Nt))
        y = chunked_pmap(lambda y, weights: poly_smooth(times, y, deg=5, weights=weights), y, weights)
        y = y.reshape((Nd * Na * (Nt // blocksize), blocksize))  # Nd*Na*(Nt//blocksize), blocksize
        return y

    logger.info("Smoothing and outlier rejection of const (a weak prior).")
    # Nd,Na,Nt/blocksize, blocksize
    const_real_mean = smooth(jnp.cos(const_mean), const_weights)  # Nd*Na*(Nt//blocksize), blocksize
    const_imag_mean = smooth(jnp.sin(const_mean), const_weights)  # Nd*Na*(Nt//blocksize), blocksize
    const_mean_smoothed = jnp.arctan2(const_imag_mean, const_real_mean)  # Nd*Na*(Nt//blocksize), blocksize

    # empirically determined uncertainty point where sigma(tec - tec_true) > 6 mTECU
    which_reprocess = jnp.any(uncert_mean > 0., axis=1)  # Nd*Na*(Nt//blocksize)
    replace_map = jnp.where(which_reprocess)

    logger.info("Performing refined tec-only solve, with fixed const.")
    keys = random.split(random.PRNGKey(int(1000 * default_timer())), jnp.sum(which_reprocess))
    # [Nd*Na*(Nt//blocksize), blocksize]
    (tec_mean_constrained, tec_std_constrained, const_mean_constrained, const_std_constrained) = \
        chunked_pmap(lambda *args: constrained_solve(freqs, *args),
                                                              keys,
                                                              phase_obs[which_reprocess],
                                                              gain_outliers[which_reprocess],
                                                              const_mean_smoothed[which_reprocess],
                                                              const_std[which_reprocess]
                                                              )
    tec_mean = tec_mean.at[replace_map].set(tec_mean_constrained)
    tec_std = tec_std.at[replace_map].set(tec_std_constrained)
    const_std = const_std.at[replace_map].set(const_std_constrained)
    const_mean = const_mean.at[replace_map].set(const_mean_constrained)

    # const_weights = 1. / const_std ** 2
    #
    # def smooth(y, weights):
    #     y = y.reshape((Nd * Na, Nt))  # Nd*Na,Nt
    #     weights = weights.reshape((Nd * Na, Nt))
    #     y = chunked_pmap(lambda y, weights: poly_smooth(times, y, deg=5, weights=weights), y, weights)
    #     y = y.reshape((Nd * Na * (Nt // blocksize), blocksize))  # Nd*Na*(Nt//blocksize), blocksize
    #     return y
    #
    # logger.info("Smoothing and outlier rejection of const (a weak prior).")
    # # Nd,Na,Nt/blocksize, blocksize
    # const_real_mean = smooth(jnp.cos(const_mean), const_weights)  # Nd*Na*(Nt//blocksize), blocksize
    # const_imag_mean = smooth(jnp.sin(const_mean), const_weights)  # Nd*Na*(Nt//blocksize), blocksize
    # const_mean = jnp.arctan2(const_imag_mean, const_real_mean)  # Nd*Na*(Nt//blocksize), blocksize

    (tec_mean, tec_std, const_mean, const_std) = tree_map(lambda x: x.reshape((Nd, Na, Nt)),
                                                          (tec_mean, tec_std, const_mean, const_std))

    # Nd, Na, Nt
    logger.info("Performing outlier detection on tec values.")
    tec_est, tec_outliers = detect_tec_outliers(times, tec_mean, tec_std)
    tec_std = jnp.where(tec_outliers, jnp.inf, tec_std)

    # remove remainder at the end
    if remainder != 0:
        (tec_mean, tec_std, const_mean, const_std) = tree_map(lambda x: x[..., :-remainder],
                                                              (tec_mean, tec_std, const_mean, const_std))

    # compute phase mean with outlier-suppressed tec.
    phase_mean = tec_mean[..., None, :] * (TEC_CONV / freqs[:, None]) + const_mean[..., None, :]
    phase_uncert = jnp.sqrt((tec_std[..., None, :] * (TEC_CONV / freqs[:, None])) ** 2 + (const_std[..., None, :]) ** 2)

    return phase_mean, phase_uncert, tec_mean, tec_std, tec_outliers, const_mean, const_std


def get_data(solution_file):
    logger.info("Getting DDS4 data.")
    with DataPack(solution_file, readonly=True) as h:
        select = dict(pol=slice(0, 1, 1))#, ant=slice(50, 51), dir=slice(0,None,1), time=slice(0, 100, 1))
        h.select(**select)
        phase, axes = h.phase
        phase = phase[0, ...]
        gain_outliers, _ = h.weights_phase
        gain_outliers = gain_outliers[0, ...] == 1
        amp, axes = h.amplitude
        amp = amp[0, ...]
        _, freqs = h.get_freqs(axes['freq'])
        freqs = freqs.to(au.Hz).value
        _, times = h.get_times(axes['time'])
        times = jnp.asarray(times.mjd, dtype=jnp.float64) * 86400.
        times = times - times[0]
        logger.info("Shape: {}".format(phase.shape))

        (Nd, Na, Nf, Nt) = phase.shape

        @jit
        def smooth(amp, outliers):
            '''
            Smooth amplitudes
            Args:
                amp: [Nt, Nf]
                outliers: [Nt, Nf]
            '''
            weights = jnp.where(outliers, 0., 1.)
            log_amp = jnp.log(amp)
            log_amp = vmap(lambda log_amp, weights: poly_smooth(times, log_amp, deg=3, weights=weights))(log_amp.T,
                                                                                                         weights.T).T
            log_amp = vmap(lambda log_amp, weights: poly_smooth(freqs, log_amp, deg=3, weights=weights))(log_amp,
                                                                                                         weights)
            amp = jnp.exp(log_amp)
            return amp

        logger.info("Smoothing amplitudes")
        amp = chunked_pmap(smooth, amp.reshape((Nd * Na, Nf, Nt)).transpose((0, 2, 1)),
                           gain_outliers.reshape((Nd * Na, Nf, Nt)).transpose((0, 2, 1)))  # Nd*Na,Nt,Nf
        amp = amp.transpose((0, 2, 1)).reshape((Nd, Na, Nf, Nt))
    return gain_outliers, phase, amp, times, freqs


def prepare_soltabs(dds4_h5parm, dds5_h5parm):
    logger.info("Creating sol000/phase000+amplitude000+tec000+const000")
    make_soltab(dds4_h5parm, from_solset='sol000', to_solset='sol000', from_soltab='phase000',
                to_soltab=['phase000', 'amplitude000', 'tec000', 'const000', 'tec_outliers000'],
                remake_solset=True,
                to_datapack=dds5_h5parm)


def main(data_dir, working_dir, obs_num, ncpu, plot_results):
    os.environ['XLA_FLAGS'] = f"--xla_force_host_platform_device_count={ncpu}"
    logger.info("Performing data smoothing via tec+const+clock inference.")
    dds4_h5parm = os.path.join(data_dir, 'L{}_DDS4_full_merged.h5'.format(obs_num))
    dds5_h5parm = os.path.join(working_dir, 'L{}_DDS5_full_merged.h5'.format(obs_num))
    linked_dds5_h5parm = os.path.join(data_dir, 'L{}_DDS5_full_merged.h5'.format(obs_num))
    logger.info("Looking for {}".format(dds4_h5parm))
    link_overwrite(dds5_h5parm, linked_dds5_h5parm)
    prepare_soltabs(dds4_h5parm, dds5_h5parm)
    gain_outliers, phase_obs, amp, times, freqs = get_data(solution_file=dds4_h5parm)
    phase_mean, phase_uncert, tec_mean, tec_std, tec_outliers, const_mean, const_std = \
        solve_and_smooth(gain_outliers, phase_obs, times, freqs)
    # exit(0)
    logger.info("Storing smoothed phase, amplitudes, tec, const, and clock")
    with DataPack(dds5_h5parm, readonly=False) as h:
        h.current_solset = 'sol000'
        # h.select(pol=slice(0, 1, 1), ant=slice(50, 51), dir=slice(0, None, 1), time=slice(0, 100, 1))
        h.select(pol=slice(0, 1, 1))
        h.phase = np.asarray(phase_mean)[None, ...]
        h.weights_phase = np.asarray(phase_uncert)[None, ...]
        h.amplitude = np.asarray(amp)[None, ...]
        h.tec = np.asarray(tec_mean)[None, ...]
        h.tec_outliers = np.asarray(tec_outliers)[None, ...]
        h.weights_tec = np.asarray(tec_std)[None, ...]
        h.const = np.asarray(const_mean)[None, ...]
        axes = h.axes_phase
        patch_names, _ = h.get_directions(axes['dir'])
        antenna_labels, _ = h.get_antennas(axes['ant'])

    if plot_results:

        diagnostic_data_dir = os.path.join(working_dir, 'diagnostic')
        os.makedirs(diagnostic_data_dir, exist_ok=True)

        logger.info("Plotting results.")
        data_plot_dir = os.path.join(working_dir, 'data_plots')
        os.makedirs(data_plot_dir, exist_ok=True)
        Nd, Na, Nf, Nt = phase_mean.shape
        for ia in range(Na):
            for id in range(Nd):

                fig, axs = plt.subplots(3, 1, sharex=True)
                axs[0].plot(times, tec_mean[id, ia, :], c='black', label='tec')
                ylim = axs[0].get_ylim()
                axs[0].vlines(times[tec_outliers[id, ia, :]], *ylim, colors='red', label='outliers', alpha=0.5)
                axs[0].set_ylim(*ylim)

                axs[1].plot(times, const_mean[id, ia, :], c='black', label='const')
                axs[1].fill_between(times, const_mean[id, ia, :] - const_std[id, ia, :],
                                    const_mean[id, ia, :] + const_std[id, ia, :],
                                    color='black', alpha=0.2)
                ylim = axs[1].get_ylim()
                axs[1].vlines(times[tec_outliers[id, ia, :]], *ylim, colors='red', label='outliers', alpha=0.5)
                axs[1].set_ylim(*ylim)

                axs[2].plot(times, tec_std[id, ia, :], c='black', label='tec_std')
                ylim = axs[2].get_ylim()
                axs[2].vlines(times[tec_outliers[id, ia, :]], *ylim, colors='red', label='outliers', alpha=0.5)
                axs[2].set_ylim(*ylim)

                axs[0].legend()
                axs[1].legend()
                axs[2].legend()

                axs[0].set_ylabel("DTEC [mTECU]")
                axs[1].set_ylabel("const [rad]")
                axs[2].set_ylabel("DTEC uncert [mTECU]")
                axs[2].set_xlabel("time [s]")

                fig.savefig(os.path.join(data_plot_dir, 'solutions_ant{:02d}_dir{:02d}.png'.format(ia, id)))
                plt.close("all")

                fig, axs = plt.subplots(4, 1, sharex=True, sharey=True)
                # phase data with input outliers
                # phase posterior with tec outliers
                # dphase with no outliers
                # phase uncertainty

                axs[0].imshow(phase_obs[id, ia, :, :], vmin=-jnp.pi, vmax=jnp.pi, cmap='twilight', aspect='auto',
                              origin='lower', interpolation='nearest')
                axs[0].imshow(jnp.where(gain_outliers[id, ia, :, :], 1., jnp.nan),
                              vmin=0., vmax=1., cmap='bone', aspect='auto',
                              origin='lower', interpolation='nearest')
                add_colorbar_to_axes(axs[0], "twilight", vmin=-jnp.pi, vmax=jnp.pi)

                axs[1].imshow(phase_mean[id, ia, :, :], vmin=-jnp.pi, vmax=jnp.pi, cmap='twilight', aspect='auto',
                              origin='lower', interpolation='nearest')
                axs[1].imshow(jnp.where(jnp.isinf(phase_uncert[id, ia, :, :]), 1., jnp.nan),
                              vmin=0., vmax=1., cmap='bone', aspect='auto',
                              origin='lower', interpolation='nearest')
                add_colorbar_to_axes(axs[1], "twilight", vmin=-jnp.pi, vmax=jnp.pi)

                dphase = wrap(wrap(phase_mean) - phase_obs)
                vmin = -0.5
                vmax = 0.5

                axs[2].imshow(dphase[id, ia, :, :], vmin=vmin, vmax=vmax, cmap='PuOr', aspect='auto',
                              origin='lower', interpolation='nearest')
                add_colorbar_to_axes(axs[2], "PuOr", vmin=vmin, vmax=vmax)

                vmin = 0.
                vmax = 0.8

                axs[3].imshow(phase_uncert[id, ia, :, :], vmin=vmin, vmax=vmax, cmap='PuOr', aspect='auto',
                              origin='lower', interpolation='nearest')
                add_colorbar_to_axes(axs[3], "PuOr", vmin=vmin, vmax=vmax)

                axs[0].set_ylabel("freq [MHz]")
                axs[1].set_ylabel("freq [MHz]")
                axs[2].set_ylabel("freq [MHz]")
                axs[3].set_ylabel("freq [MHz]")
                axs[3].set_xlabel("time [s]")

                axs[0].set_title("phase data [rad]")
                axs[1].set_title("phase model [rad]")
                axs[2].set_title("phase diff. [rad]")
                axs[3].set_title("phase uncert [rad]")

                fig.savefig(os.path.join(data_plot_dir, 'data_comparison_ant{:02d}_dir{:02d}.png'.format(ia, id)))
                plt.close("all")
        # exit(0)

        d = os.path.join(working_dir, 'tec_plots')
        animate_datapack(dds5_h5parm, d, num_processes=(ncpu * 2) // 3,
                         vmin=-60,
                         vmax=60., observable='tec', phase_wrap=False, plot_crosses=False,
                         plot_facet_idx=True, labels_in_radec=True, per_timestep_scale=True,
                         solset='sol000', cmap=plt.cm.PuOr)
        # os.makedirs(d, exist_ok=True)
        # DatapackPlotter(dds5_h5parm).plot(
        #     fignames=[os.path.join(d, "fig-{:04d}.png".format(j)) for j in range(Nt)],
        #     vmin=-60,
        #     vmax=60., observable='tec', phase_wrap=False, plot_crosses=False,
        #     plot_facet_idx=True, labels_in_radec=True, per_timestep_scale=True,
        #     solset='sol000', cmap=plt.cm.PuOr)
        # make_animation(d, prefix='fig', fps=4)

        d = os.path.join(working_dir, 'const_plots')
        animate_datapack(dds5_h5parm, d, num_processes=(ncpu * 2) // 3,
                         vmin=-np.pi,
                         vmax=np.pi, observable='const', phase_wrap=False, plot_crosses=False,
                         plot_facet_idx=True, labels_in_radec=True, per_timestep_scale=True,
                         solset='sol000', cmap=plt.cm.PuOr)

        # os.makedirs(d, exist_ok=True)
        # DatapackPlotter(dds5_h5parm).plot(
        #     fignames=[os.path.join(d, "fig-{:04d}.png".format(j)) for j in range(Nt)],
        #     vmin=-np.pi,
        #     vmax=np.pi, observable='const', phase_wrap=False, plot_crosses=False,
        #     plot_facet_idx=True, labels_in_radec=True, per_timestep_scale=True,
        #     solset='sol000', cmap=plt.cm.PuOr)
        # make_animation(d, prefix='fig', fps=4)

        d = os.path.join(working_dir, 'clock_plots')
        animate_datapack(dds5_h5parm, d, num_processes=(ncpu * 2) // 3,
                         vmin=None,
                         vmax=None,
                         observable='clock', phase_wrap=False, plot_crosses=False,
                         plot_facet_idx=True, labels_in_radec=True, per_timestep_scale=True,
                         solset='sol000', cmap=plt.cm.PuOr)

        # os.makedirs(d, exist_ok=True)
        # DatapackPlotter(dds5_h5parm).plot(
        #     fignames=[os.path.join(d, "fig-{:04d}.png".format(j)) for j in range(Nt)],
        #     vmin=None,
        #     vmax=None,
        #     observable='clock', phase_wrap=False, plot_crosses=False,
        #     plot_facet_idx=True, labels_in_radec=True, per_timestep_scale=True,
        #     solset='sol000', cmap=plt.cm.PuOr)
        # make_animation(d, prefix='fig', fps=4)

        d = os.path.join(working_dir, 'amplitude_plots')
        animate_datapack(dds5_h5parm, d, num_processes=(ncpu * 2) // 3,
                         log_scale=True, observable='amplitude', phase_wrap=False, plot_crosses=False,
                         plot_facet_idx=True, labels_in_radec=True, per_timestep_scale=True,
                         solset='sol000', cmap=plt.cm.PuOr
                         )
        # os.makedirs(d, exist_ok=True)
        # DatapackPlotter(dds5_h5parm).plot(
        #     fignames=[os.path.join(d, "fig-{:04d}.png".format(j)) for j in range(Nt)],
        #     log_scale=True, observable='amplitude', phase_wrap=False, plot_crosses=False,
        #     plot_facet_idx=True, labels_in_radec=True, per_timestep_scale=True,
        #     solset='sol000', cmap=plt.cm.PuOr)
        # make_animation(d, prefix='fig', fps=4)


def debug_main():
    main(obs_num=342938,
         data_dir="/home/albert/data/gains_screen/data",
         working_dir="/home/albert/data/gains_screen/data",
         ncpu=8,
         plot_results=True)


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
