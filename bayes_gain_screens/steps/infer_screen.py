import matplotlib

# matplotlib.use('Agg')

import argparse
import os
from timeit import default_timer
import numpy as np
import pylab as plt
import astropy.units as au

from jax import random, vmap, numpy as jnp, tree_map, jit
from jax.scipy.linalg import solve_triangular
from jax.scipy.special import logsumexp
from jax.lax import while_loop

from bayes_gain_screens.utils import chunked_pmap, get_screen_directions_from_image, link_overwrite, polyfit
from bayes_gain_screens.plotting import make_animation, DatapackPlotter, plot_vornoi_map

from h5parm import DataPack
from h5parm.utils import make_soltab

from jaxns.gaussian_process.kernels import RBF, M12, M32, M52, RationalQuadratic
from jaxns.prior_transforms import UniformPrior, PriorChain, HalfLaplacePrior, GaussianProcessKernelPrior, \
    DeterministicTransformPrior
from jaxns.nested_sampling import NestedSampler
from jaxns.utils import resample, left_broadcast_mul
from jaxns.plotting import plot_cornerplot, plot_diagnostics

import sys
import logging

logger = logging.getLogger(__name__)

TEC_CONV = -8.4479745e6  # mTECU/Hz

def log_normal_outliers(x, mean, cov, sigma):
    """
    Computes log-Normal density with outliers removed.

    Args:
        x: RV value
        mean: mean of Gaussian
        cov: covariance of underlying, minus the obs. covariance
        sigma: stddev's of obs. error, inf encodes an outlier.

    Returns: a normal density for all points not of inf stddev obs. error.
    """
    C = cov / (sigma[:, None] * sigma[None, :]) + jnp.eye(cov.shape[0])
    L = jnp.linalg.cholesky(C)
    Ls = sigma[:, None] * L
    log_det = jnp.sum(jnp.where(jnp.isinf(sigma), 0., jnp.log(jnp.diag(Ls))))
    dx = (x - mean)
    dx = solve_triangular(L, dx / sigma, lower=True)
    maha = dx @ dx
    log_likelihood = -0.5 * jnp.sum(~jnp.isinf(sigma)) * jnp.log(2. * jnp.pi) \
                     - log_det \
                     - 0.5 * maha
    return log_likelihood


def marginalise_static(key, samples, log_weights, ESS, fun):
    """
    Marginalises function over posterior samples, where ESS is static.

    Args:
        key: PRNG key
        samples: dict of batched array of nested sampling samples
        log_weights: log weights from nested sampling
        ESS: static effective sample size
        fun: callable(**kwargs) to marginalise.

    Returns: expectation over resampled samples.
    """
    samples = resample(key, samples, log_weights, S=ESS)
    marginalised = jnp.mean(vmap(lambda d: fun(**d))(samples), axis=0)
    return marginalised


def marginalise(key, samples, log_weights, ESS, fun):
    """
    Marginalises function over posterior samples, where ESS can be dynamic.

    Args:
        key: PRNG key
        samples: dict of batched array of nested sampling samples
        log_weights: log weights from nested sampling
        ESS: dynamic effective sample size
        fun: callable(**kwargs) to marginalise.

    Returns: expectation over resampled samples.
    """

    def body(state):
        (key, i, marginalised) = state
        key, resample_key = random.split(key, 2)
        _samples = tree_map(lambda v: v[0], resample(resample_key, samples, log_weights, S=1))
        marginalised += fun(**_samples)
        return (key, i + 1., marginalised)

    test_output = fun(**tree_map(lambda v: v[0], samples))
    (_, count, marginalised) = while_loop(lambda state: state[1] < ESS,
                                          body,
                                          (key, jnp.array(0.), jnp.zeros_like(test_output)))
    marginalised = marginalised / count
    return marginalised

@jit
def nn_interp(x,y,xstar):
    def single_interp(xstar):
        dx = jnp.linalg.norm(xstar - x, axis=1)
        dx = jnp.where(dx == 0., jnp.inf, dx)
        imin = jnp.argmin(dx)
        return y[imin]
    return vmap(single_interp)(xstar)

@jit
def nn_smooth(x,y,xstar):
    def single_interp(xstar):
        dx = jnp.linalg.norm(xstar - x, axis=1)
        nn_dist = jnp.min(jnp.where(dx == 0., jnp.inf, dx))
        dx = dx/nn_dist
        weight = jnp.exp(-0.5*dx**2)
        weight /= jnp.sum(weight)
        return jnp.sum(y*weight)
    return vmap(single_interp)(xstar)

def single_screen(key, amp, tec_mean, tec_std, const, clock, directions, screen_directions, freqs):
    """
    Computes a screen over `screen_directions` conditioned on data in `directions`.
    Uses a 3-rd order polynomial in frequency approximation for amplitude.
    Computes a screen for const on the unit circle using a Cartesian projection.

    Args:
        key: PRNG key
        amp: [Nd, Nf] amplitudes
        tec_mean: [Nd] tec estimates
        tec_std: [Nd] tec uncert estimates with jnp.inf meaning outlier (might be extra remaining outliers)
        const: [Nd] constant phase estimates
        clock: [Nd] clock estimates
        directions: [Nd, 2] directions of data in radians
        screen_directions: [Nd_screen, 2] directions of screen points in radians
        freqs: [Nf] freqs in Hz

    Returns: tuple of posterior means
        post_phase
        post_amp
        post_tec
        post_const
        post_clock
    """
    Nd = directions.shape[0]
    Nd_screen = screen_directions.shape[0]

    post_amp = vmap(lambda amp: nn_smooth(directions,amp,screen_directions))(amp.T).T
    post_const = nn_smooth(directions, const, screen_directions)
    post_clock = nn_smooth(directions, clock ,screen_directions)
    # post_tec = nn_smooth(directions, tec_mean, screen_directions)

    def _min_dist(_direction):
        dist = jnp.linalg.norm(_direction - directions, axis=1)
        return jnp.min(jnp.where(dist == 0., jnp.inf, dist))

    minimum_dist = jnp.min(vmap(_min_dist)(directions))
    avg_dist2 = jnp.mean(vmap(_min_dist)(directions) ** 2)

    def _max_dist(_direction):
        dist = jnp.linalg.norm(_direction - directions, axis=1)
        return jnp.max(dist)

    maximum_dist = jnp.max(vmap(_max_dist)(screen_directions))

    def _nearest_diff(_v, _direction, v):
        dist = jnp.linalg.norm(_direction - directions, axis=1)
        idx = jnp.argmin(jnp.where(dist == 0., jnp.inf, dist))
        return _v - v[idx]

    def build_sigma_est(v):
        sigma2_est = jnp.mean(
            jnp.square(vmap(lambda _v, _directions: _nearest_diff(_v, _directions, v))(v, directions)))
        sigma2_min = sigma2_est * jnp.exp(0.5 * avg_dist2 / maximum_dist ** 2)
        sigma2_max = sigma2_est * jnp.exp(0.5 * avg_dist2 / minimum_dist ** 2)
        return jnp.sqrt(sigma2_min), jnp.sqrt(sigma2_max)


    def gp_inference(key, v, obs_uncert, kernel):
        # normalise
        v_mean = jnp.mean(v)
        v_std = jnp.maximum(jnp.std(v), 1e-6)
        v = (v - v_mean) / v_std
        obs_uncert = obs_uncert / v_std
        # protect against situation of singular evidence distribution
        v = v + 0.001*random.normal(key,shape=v.shape)

        l = HalfLaplacePrior('l', jnp.sqrt(avg_dist2), tracked=True)
        sigma = HalfLaplacePrior('sigma', 1.)#build_sigma_est(v)[1])#UniformPrior('sigma', *build_sigma_est(v))#
        uncert = HalfLaplacePrior('_uncert', 0.05)
        uncert = DeterministicTransformPrior('uncert',
                                             lambda _uncert: jnp.sqrt(obs_uncert ** 2 + _uncert ** 2),
                                             obs_uncert.shape, uncert, tracked=True)

        kernel_params = [l, sigma]

        if kernel.__class__.__name__ == 'RationalQuadratic':
            alpha = UniformPrior('alpha', 0.5, 50.)
            kernel_params.append(alpha)

        K = GaussianProcessKernelPrior('K',
                                       kernel,
                                       directions,
                                       *kernel_params,
                                       tracked=True)
        kernel_param_names = [p.name for p in kernel_params]

        def predict_f(K, uncert, **kwargs):
            # (K + sigma.sigma)^-1 = sigma^-1.(sigma^-1.K.sigma^-1 + I)^-1.sigma^-1
            kernel_params = [kwargs.get(n) for n in kernel_param_names]
            Kstar = kernel(directions, screen_directions, *kernel_params)
            C = K / (uncert[:, None] * uncert[None, :]) + jnp.eye(K.shape[0])
            JT = jnp.linalg.solve(C, Kstar / uncert[:, None])
            return (JT.T @ (v / uncert)) * v_std + v_mean

        def predict_fvar(K, uncert, **kwargs):
            # (K + sigma.sigma)^-1 = sigma^-1.(sigma^-1.K.sigma^-1 + I)^-1.sigma^-1
            sigma = kwargs.get('sigma')
            kernel_params = [kwargs.get(n) for n in kernel_param_names]
            Kstar = kernel(directions, screen_directions, *kernel_params)
            C = K / (uncert[:, None] * uncert[None, :]) + jnp.eye(K.shape[0])
            JT = jnp.linalg.solve(C, Kstar / uncert[:, None])
            return (sigma ** 2 * jnp.eye(Nd_screen) - jnp.diag(JT.T @ (Kstar / uncert[:, None]))) * v_std ** 2

        def log_likelihood(K, uncert, **kwargs):
            return log_normal_outliers(v, 0., K, uncert)

        prior_chain = PriorChain().push(K).push(uncert)

        # print(prior_chain)

        ns = NestedSampler(log_likelihood,
                           prior_chain,
                           sampler_name='slice'
                           )

        key, key_ns = random.split(key,2)
        results = ns(key=key,
                     num_live_points=100,
                     max_samples=5e4,
                     collect_samples=True,
                     only_marginalise=False,
                     termination_frac=0.001,
                     sampler_kwargs=dict(depth=2, num_slices=4))


        key, key_post_f, key_post_fvar = random.split(key, 3)
        post_f = marginalise_static(key_post_f, results.samples, results.log_p, 250, predict_f)
        # post_fvar = marginalise_static(key_post_fvar, results.samples, results.log_p, 500, predict_fvar)
        return post_f, results.logZ, results


    key, key_tec = random.split(key, 2)
    kernels = [RationalQuadratic()]
    @jit
    def gp_smooth(key, v, obs_uncert):
        keys = random.split(key, len(kernels))
        post_f, logZ, results = [],[], []
        for key, kernel in zip(keys, kernels):
            _post_f, _logZ, _results = gp_inference(key, v, obs_uncert, kernel)
            post_f.append(_post_f)
            logZ.append(_logZ)
            results.append(_results)
        logZ = jnp.asarray(logZ)
        post_f = jnp.stack(post_f, axis=0)
        weights = jnp.exp(logZ - logsumexp(logZ))
        post_f = jnp.sum(left_broadcast_mul(weights, post_f), axis=0)
        return post_f, weights, results

    # logger.info("Performing tec inference")
    post_tec, weights, results = gp_smooth(key_tec, tec_mean, tec_std)
    # logger.info("Weights: {}".format(weights))
    # post_tec.block_until_ready()
    # # t0 = default_timer()
    # # post_tec, weights, results = gp_smooth(key_tec, tec_mean, tec_std)
    # # post_tec.block_until_ready()
    # # print(default_timer() - t0)
    #
    # plot_diagnostics(results[0])
    # plot_cornerplot(results[0], vars=['l', 'sigma', '_uncert', 'alpha'])

    # plot_vornoi_map(directions, colors=tec_mean, cmap=plt.cm.PuOr, relim=True)
    # plt.show()
    # plot_vornoi_map(screen_directions, colors=post_tec, cmap=plt.cm.PuOr, relim=True)
    # plt.show()
    #
    # plot_vornoi_map(directions, colors=const, cmap=plt.cm.hsv, vmin=-np.pi, vmax=np.pi, relim=True)
    # plt.show()
    # plot_vornoi_map(screen_directions, colors=post_const,cmap=plt.cm.hsv,vmin=-np.pi, vmax=np.pi, relim=True)
    # plt.show()
    #
    # plot_vornoi_map(directions, colors=clock, cmap=plt.cm.PuOr, relim=True)
    # plt.show()
    # plot_vornoi_map(screen_directions, colors=post_clock, cmap=plt.cm.PuOr, relim=True)
    # plt.show()
    #
    # plot_vornoi_map(directions, colors=amp[:,12], vmin=0.5, vmax=1.5, relim=True)
    # plt.show()
    # plot_vornoi_map(screen_directions, colors=post_amp[:, 12], vmin=0.5, vmax=1.5, relim=True)
    # plt.show()

    post_phase = post_tec[:, None] * (TEC_CONV / freqs) \
                 + post_const[:, None] \
                 + post_clock[:, None] * (2. * jnp.pi * 1e-9 * freqs)

    return post_phase, post_amp, post_tec, post_const, post_clock


def screen_model(amp, tec_mean, tec_std, const, clock, directions, screen_directions, freqs):
    Nd_screen = screen_directions.shape[0]
    Nd, Na, Nf, Nt = amp.shape
    tec_mean = tec_mean.transpose((1, 2, 0)).reshape((Na * Nt, Nd))
    tec_std = tec_std.transpose((1, 2, 0)).reshape((Na * Nt, Nd))
    const = const.transpose((1, 2, 0)).reshape((Na * Nt, Nd))
    clock = clock.transpose((1, 2, 0)).reshape((Na * Nt, Nd))
    amp = amp.transpose((1, 3, 0, 2)).reshape((Na * Nt, Nd, Nf))

    T = Na * Nt
    keys = random.split(random.PRNGKey(int(default_timer())), T)

    # m = 0
    # single_screen(keys[m], amp[m], tec_mean[m], tec_std[m], const[m], clock[m], directions, screen_directions, freqs)
    # return

    post_phase, post_amp, post_tec, post_const, post_clock = \
        chunked_pmap(lambda key, amp, tec_mean, tec_std, const, clock:
                     single_screen(key, amp, tec_mean, tec_std, const, clock, directions, screen_directions, freqs),
                     keys, amp, tec_mean, tec_std, const, clock, debug_mode=False, chunksize=None)

    post_phase = post_phase.reshape((Na, Nt, Nd_screen, Nf)).transpose((2, 0, 3, 1))
    post_amp = post_amp.reshape((Na, Nt, Nd_screen, Nf)).transpose((2, 0, 3, 1))
    post_tec = post_tec.reshape((Na, Nt, Nd_screen)).transpose((2, 0, 1))
    post_const = post_const.reshape((Na, Nt, Nd_screen)).transpose((2, 0, 1))
    post_clock = post_clock.reshape((Na, Nt, Nd_screen)).transpose((2, 0, 1))

    return post_phase, post_amp, post_tec, post_const, post_clock


def prepare_soltabs(dds5_h5parm, dds6_h5parm, screen_directions):
    logger.info("Creating sol000/phase000+amplitude000+tec000+const000+clock000")
    make_soltab(dds5_h5parm, from_solset='sol000', to_solset='sol000', from_soltab='phase000',
                to_soltab=['phase000', 'amplitude000', 'tec000', 'const000', 'clock000'],
                remake_solset=True,
                to_datapack=dds6_h5parm,
                directions=screen_directions)


def generate_data(dds5_h5parm):
    with DataPack(dds5_h5parm, readonly=True) as h:
        select = dict(pol=slice(0, 1, 1), ant=51, time=slice(0,128,1))
        h.current_solset = 'sol000'
        h.select(**select)
        amp, axes = h.amplitude
        amp = amp[0, ...]
        phase, axes = h.phase
        phase = phase[0, ...]
        _, freqs = h.get_freqs(axes['freq'])
        freqs = freqs.to(au.Hz).value
        tec_mean, axes = h.tec
        tec_mean = tec_mean[0, ...]
        tec_std, axes = h.weights_tec
        tec_outliers, _ = h.tec_outliers
        tec_std = jnp.where(tec_outliers == 1., jnp.inf, tec_std)
        tec_std = tec_std[0, ...]
        const, _ = h.const
        const = const[0, ...]
        clock, _ = h.clock
        clock = clock[0, ...]
        patch_names, directions = h.get_directions(axes['dir'])
        directions = jnp.stack([directions.ra.rad, directions.dec.rad], axis=-1)
    return phase, amp, tec_mean, tec_std, const, clock, directions, freqs


def main(data_dir, working_dir, obs_num, ref_image_fits, ncpu, max_N, plot_results):
    os.environ['XLA_FLAGS'] = f"--xla_force_host_platform_device_count={ncpu}"
    # os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false "
    #                            "intra_op_parallelism_threads=1")

    dds5_h5parm = os.path.join(data_dir, 'L{}_DDS5_full_merged.h5'.format(obs_num))
    dds6_h5parm = os.path.join(working_dir, 'L{}_DDS6_full_merged.h5'.format(obs_num))
    linked_dds6_h5parm = os.path.join(data_dir, 'L{}_DDS6_full_merged.h5'.format(obs_num))
    logger.info("Looking for {}".format(dds5_h5parm))
    link_overwrite(dds6_h5parm, linked_dds6_h5parm)

    phase, amp, tec_mean, tec_std, const, clock, directions, freqs = generate_data(dds5_h5parm)
    Nd, Na, Nf, Nt = amp.shape

    screen_directions, _ = get_screen_directions_from_image(ref_image_fits,
                                                            flux_limit=0.01,
                                                            max_N=max_N,
                                                            min_spacing_arcmin=4.,
                                                            seed_directions=directions,
                                                            fill_in_distance=8.,
                                                            fill_in_flux_limit=0.01)

    prepare_soltabs(dds5_h5parm, dds6_h5parm, screen_directions)

    screen_directions = jnp.stack([screen_directions.ra.rad, screen_directions.dec.rad], axis=-1)

    post_phase, post_amp, post_tec, post_const, post_clock = screen_model(amp, tec_mean, tec_std, const, clock,
                                                                          directions, screen_directions, freqs)

    with DataPack(dds6_h5parm, readonly=False) as h:
        h.current_solset = 'sol000'
        h.select(pol=slice(0, 1, 1), ant=51, time=slice(0,128,1))
        h.tec = np.asarray(post_tec)[None, ...]
        h.const = np.asarray(post_const)[None, ...]
        h.clock = np.asarray(post_clock)[None, ...]
        h.amplitude = np.asarray(post_amp)[None, ...]
        h.phase = np.asarray(post_phase)[None, ...]
        h.select(pol=slice(0, 1, 1), dir=slice(0, Nd, 1))
        h.phase = np.asarray(phase)[None, ...]
        h.amplitude = np.asarray(amp)[None, ...]

    if plot_results:
        logger.info("Plotting results.")
        d = os.path.join(working_dir, 'tec_screen_plots')
        os.makedirs(d, exist_ok=True)
        DatapackPlotter(dds6_h5parm).plot(
            fignames=[os.path.join(d, "fig-{:04d}.png".format(j)) for j in range(Nt)],
            vmin=-60,
            vmax=60., observable='tec', phase_wrap=False, plot_crosses=False,
            plot_facet_idx=False, labels_in_radec=True, per_timestep_scale=True,
            solset='sol000', cmap=plt.cm.PuOr)
        make_animation(d, prefix='fig', fps=4)

        d = os.path.join(working_dir, 'const_screen_plots')
        os.makedirs(d, exist_ok=True)
        DatapackPlotter(dds6_h5parm).plot(
            fignames=[os.path.join(d, "fig-{:04d}.png".format(j)) for j in range(Nt)],
            vmin=-np.pi,
            vmax=np.pi, observable='const', phase_wrap=False, plot_crosses=False,
            plot_facet_idx=False, labels_in_radec=True, per_timestep_scale=True,
            solset='sol000', cmap=plt.cm.PuOr)
        make_animation(d, prefix='fig', fps=4)

        d = os.path.join(working_dir, 'clock_screen_plots')
        os.makedirs(d, exist_ok=True)
        DatapackPlotter(dds6_h5parm).plot(
            fignames=[os.path.join(d, "fig-{:04d}.png".format(j)) for j in range(Nt)],
            vmin=None,
            vmax=None,
            observable='clock', phase_wrap=False, plot_crosses=False,
            plot_facet_idx=False, labels_in_radec=True, per_timestep_scale=True,
            solset='sol000', cmap=plt.cm.PuOr)
        make_animation(d, prefix='fig', fps=4)

        d = os.path.join(working_dir, 'amplitude_screen_plots')
        os.makedirs(d, exist_ok=True)
        DatapackPlotter(dds6_h5parm).plot(
            fignames=[os.path.join(d, "fig-{:04d}.png".format(j)) for j in range(Nt)],
            log_scale=True, observable='amplitude', phase_wrap=False, plot_crosses=False,
            plot_facet_idx=False, labels_in_radec=True, per_timestep_scale=True,
            solset='sol000', cmap=plt.cm.PuOr)
        make_animation(d, prefix='fig', fps=4)


def debug_main():
    main(data_dir='/home/albert/data/gains_screen/data',
         working_dir='/home/albert/data/gains_screen/working_dir/',
         obs_num=342938,
         ref_image_fits='/home/albert/data/gains_screen/data/lotss_archive_deep_image.app.restored.fits',
         ncpu=2,
         max_N=250,
         plot_results=True)


def add_args(parser):
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument('--obs_num', help='Obs number L*',
                        default=None, type=int, required=True)
    parser.add_argument('--data_dir', help='Where are the ms files are stored.',
                        default=None, type=str, required=True)
    parser.add_argument('--working_dir', help='Where to perform the imaging.',
                        default=None, type=str, required=True)
    parser.add_argument('--ncpu', help='Number of CPUs.',
                        default=32, type=int, required=True)
    parser.add_argument('--max_N', help='The maximum number of screen directions.',
                        default=250, type=int, required=False)
    parser.add_argument('--ref_image_fits',
                        help='The Gaussian source list of the field used to choose locations of screen points.',
                        type=str, required=True)
    parser.add_argument('--plot_results', help='Whether to plot results.',
                        default=False, type="bool", required=False)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        debug_main()
        exit(0)
    parser = argparse.ArgumentParser(
        description='Infers the value of DTEC and a constant over a screen.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_args(parser)
    flags, unparsed = parser.parse_known_args()
    logger.info("Running with:")
    for option, value in vars(flags).items():
        logger.info("\t{} -> {}".format(option, value))
    main(**vars(flags))
