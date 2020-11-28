import os
import numpy as np
import pylab as plt
import argparse
import sys
from timeit import default_timer
from jax import numpy as jnp, vmap, jit, local_device_count, tree_multimap, pmap, tree_map, random
import logging
import astropy.units as au

logger = logging.getLogger(__name__)

from bayes_gain_screens import logging
from bayes_gain_screens.plotting import animate_datapack

from h5parm import DataPack
from h5parm.utils import make_soltab

from jaxns.prior_transforms import UniformPrior, PriorChain, HalfLaplacePrior, DeterministicTransformPrior
from jaxns.nested_sampling import NestedSampler


def polyfit(x, y, deg):
    """
    x : array_like, shape (M,)
        x-coordinates of the M sample points ``(x[i], y[i])``.
    y : array_like, shape (M,) or (M, K)
        y-coordinates of the sample points. Several data sets of sample
        points sharing the same x-coordinates can be fitted at once by
        passing in a 2D-array that contains one dataset per column.
    deg : int
        Degree of the fitting polynomial
    Returns
    -------
    p : ndarray, shape (deg + 1,) or (deg + 1, K)
        Polynomial coefficients, highest power first.  If `y` was 2-D, the
        coefficients for `k`-th data set are in ``p[:,k]``.
    """
    order = int(deg) + 1
    if deg < 0:
        raise ValueError("expected deg >= 0")
    if x.ndim != 1:
        raise TypeError("expected 1D vector for x")
    if x.size == 0:
        raise TypeError("expected non-empty vector for x")
    if y.ndim < 1 or y.ndim > 2:
        raise TypeError("expected 1D or 2D array for y")
    if x.shape[0] != y.shape[0]:
        raise TypeError("expected x and y to have same length")
    rcond = len(x) * jnp.finfo(x.dtype).eps
    lhs = jnp.stack([x ** (deg - i) for i in range(order)], axis=1)
    rhs = y
    scale = jnp.sqrt(jnp.sum(lhs * lhs, axis=0))
    lhs /= scale
    c, resids, rank, s = jnp.linalg.lstsq(lhs, rhs, rcond)
    c = (c.T / scale).T  # broadcast scale coefficients
    return c


def poly_smooth(x, y, deg=5):
    """
    Smooth y(x) with a `deg` degree polynomial in x
    Args:
        x: [N]
        y: [N]
        deg: int

    Returns: smoothed y [N]
    """
    coeffs = polyfit(x, y, deg=deg)
    return sum([p * x ** (deg - i) for i, p in enumerate(coeffs)])


def chunked_pmap(f, *args, chunksize=None):
    """
    Calls pmap on chunks of moderate work to be distributed over devices.
    Automatically handle non-dividing chunksizes, by adding filler elements.
    
    Args:
        f: callable
        *args: arguments to map down first dimension
        chunksize: optional chunk size else num devices

    Returns: pytree mapped result.
    """
    if chunksize is None:
        chunksize = local_device_count()
    N = len(args[0])
    remainder = N % chunksize
    if (remainder != 0) and (N > chunksize):
        args = [jnp.concatenate([arg, arg[:remainder]], axis=0) for arg in args]
        N = len(args[0])
    logger.info("Running on ", chunksize)
    results = []
    for start in range(0, N, chunksize):
        stop = min(start + chunksize, N)
        t0 = default_timer()
        results.append(pmap(f)(*[arg[start:stop] for arg in args]))
        # if isinstance(results,(tuple,list)):
        #     results[-1][0].block_until_ready()
        # else:
        #     results[-1].block_until_ready()
        logger.info("Time:", default_timer() - t0)
    result = tree_multimap(lambda *args: jnp.concatenate(args, axis=0), *results)
    if remainder != 0:
        result = tree_map(lambda x: x[:-remainder], result)
    return result


def get_data(solution_file):
    with DataPack(solution_file, readonly=True) as h:
        select = dict(pol=slice(0, 1, 1))
        h.select(**select)
        phase, axes = h.phase
        phase = phase[0, ...]
        amp, axes = h.amplitude
        amp = amp[0, ...]
        _, freqs = h.get_freqs(axes['freq'])
        freqs = freqs.to(au.Hz).value
        _, times = h.get_times(axes['time'])
        times = times.mjd / 86400.
        logger.info("Shape:", phase.shape)

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

        amp = chunked_pmap(smooth, amp.reshape((Nd * Na, Nf, Nt)).transpose((0, 2, 1)))  # Nd*Na,Nt,Nf
        amp = amp.transpose((0, 2, 1)).reshape((Nd, Na, Nf, Nt))
        Y_obs = jnp.concatenate([amp * jnp.cos(phase), amp * jnp.sin(phase)], axis=1)
    return Y_obs, times, freqs


def prepare_soltabs(solution_file):
    with DataPack(solution_file, readonly=False) as datapack:
        logger.info("Creating sol001/phase000+amplitude000+tec000+const000+clock000")
        make_soltab(datapack, from_solset='sol000', to_solset='sol001', from_soltab='phase000',
                    to_soltab=['phase000', 'amplitude000', 'tec000', 'const000', 'clock000'], remake_solset=True)


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
                       const_mean=lambda const, **kwargs: jnp.exp(1j * const),
                       clock_mean=lambda clock, **kwargs: clock,
                       )

    results = ns(key=key,
                 num_live_points=100,
                 max_samples=1e5,
                 collect_samples=False,
                 termination_frac=0.01,
                 sampler_kwargs=dict(depth=1, num_slices=3))
    const_mean = jnp.angle(results.marginalised['const_mean'])
    clock_mean = results.marginalised['clock_mean']
    return (const_mean, clock_mean)


def constrained_solve(freqs, key, Y_obs, amp, const_mu, clock_mu):
    TEC_CONV = -8.4479745e6  # mTECU/Hz

    def log_likelihood(Y, uncert, **kwargs):
        return jnp.sum(log_laplace(Y, Y_obs, uncert))

    tec = UniformPrior('tec', -300., 300.)

    def Y_transform(tec):
        phase = tec * (TEC_CONV / freqs) + const_mu + clock_mu * 1e-9 * (2. * jnp.pi * freqs)
        return jnp.concatenate([amp * jnp.cos(phase), amp * jnp.sin(phase)])

    prior_chain = PriorChain() \
        .push(tec) \
        .push(HalfLaplacePrior('uncert', 0.2)) \
        .push(DeterministicTransformPrior('Y', Y_transform, (freqs.size * 2,), tec))

    ns = NestedSampler(log_likelihood, prior_chain,
                       sampler_name='slice',
                       tec_mean=lambda tec, **kwargs: tec,
                       tec2_mean=lambda tec, **kwargs: tec ** 2,
                       Y_mean=lambda Y, **kwargs: Y,
                       Y2_mean=lambda Y, **kwargs: Y ** 2
                       )

    results = ns(key=key,
                 num_live_points=100,
                 max_samples=1e5,
                 collect_samples=False,
                 termination_frac=0.01,
                 sampler_kwargs=dict(depth=1, num_slices=3))

    tec_mean = results.marginalised['tec_mean']
    tec_std = jnp.sqrt(results.marginalised['tec2_mean'] - results.marginalised['tec_mean'] ** 2)
    Y_mean = results.marginalised['Y_mean']
    Y_std = jnp.sqrt(results.marginalised['Y2_mean'] - results.marginalised['Y_mean'] ** 2)

    return (tec_mean, tec_std, Y_mean, Y_std)


def solve_and_smooth(Y_obs, times, freqs):
    Nd, Na, Nf, Nt = Y_obs.shape
    Y_obs = Y_obs.transpose((0, 1, 3, 2)).reshape((Nd * Na * Nt, Nf))  # Nd*Na*Nt, Nf
    amp = jnp.sqrt(Y_obs[:, :freqs.size] ** 2 + Y_obs[:, freqs.size:] ** 2)
    T = Y_obs.shape[0]
    const_mean, clock_mean = chunked_pmap(lambda *args: unconstrained_solve(freqs, *args),
                                          random.split(random.PRNGKey(int(default_timer())), T),
                                          Y_obs, amp)

    def smooth(y):
        y = y.reshape((Nd * Na, Nt))  # Nd*Na,Nt
        y = chunked_pmap(lambda y: poly_smooth(times, y, deg=3), y).reshape(
            (Nd * Na * Nt,))  # Nd*Na*Nt
        return y

    # Nd*Na*Nt
    clock_mean = smooth(clock_mean)
    const_mean = smooth(const_mean)

    (tec_mean, tec_std, Y_mean, Y_std) = \
        chunked_pmap(lambda *args: constrained_solve(freqs, *args),
                     random.split(random.PRNGKey(int(default_timer())), T), Y_obs,
                     amp, const_mean, clock_mean)
    Y_mean = Y_mean.reshape((Nd,Na,Nt, Nf)).transpose((0, 1,3,2))
    Y_std = Y_std.reshape((Nd,Na,Nt, Nf)).transpose((0, 1,3,2))
    tec_mean = tec_mean.reshape((Nd,Na,Nt))
    tec_std = tec_std.reshape((Nd,Na,Nt))
    const_mean = const_mean.reshape((Nd,Na,Nt))
    clock_mean = clock_mean.reshape((Nd,Na,Nt))

    return Y_mean, Y_std, tec_mean, tec_std, const_mean, clock_mean


def main(data_dir, working_dir, obs_num, ncpu):
    os.environ['XLA_FLAGS'] = f"--xla_force_host_platform_device_count={ncpu}"
    logger.info("Performing data smoothing via tec+const+clock inference.")
    merged_h5parm = os.path.join(data_dir, 'L{}_DDS4_full_merged.h5'.format(obs_num))
    logger.info("Working on {}".format(merged_h5parm))
    prepare_soltabs(solution_file=merged_h5parm)
    Y_obs, times, freqs = get_data(solution_file=merged_h5parm)
    Y_mean, Y_std, tec_mean, tec_std, const_mean, clock_mean = solve_and_smooth(Y_obs, times, freqs)
    phase_mean = jnp.arctan2(Y_mean[:,:,freqs.size:,:], Y_mean[:,:,:freqs.size,:])
    amp_mean = jnp.sqrt(Y_mean[:,:,freqs.size:,:]**2 + Y_mean[:,:,:freqs.size,:]**2)
    logger.info("Storing smoothed phase, amplitudes, tec, const, and clock")
    with DataPack(merged_h5parm, readonly=False) as h:
        h.current_solset = 'sol001'
        h.select(pol=slice(0, 1, 1))
        h.phase = phase_mean
        h.amplitude = amp_mean
        h.tec = tec_mean
        h.weights_tec = tec_std
        h.const = const_mean
        h.clock = clock_mean
    
    logger.info("Plotting results.")

    animate_datapack(merged_h5parm, os.path.join(working_dir, 'tec_plots'), num_processes=ncpu,
                     solset='sol001',
                     observable='tec', vmin=-60., vmax=60., plot_facet_idx=True,
                     labels_in_radec=True, plot_crosses=False, phase_wrap=False,
                     flag_outliers=False)

    animate_datapack(merged_h5parm, os.path.join(working_dir, 'const_plots'), num_processes=ncpu,
                     solset='sol001',
                     observable='const', vmin=-np.pi, vmax=np.pi, plot_facet_idx=True,
                     labels_in_radec=True, plot_crosses=False, phase_wrap=True,
                     flag_outliers=False)

    animate_datapack(merged_h5parm, os.path.join(working_dir, 'clock_plots'), num_processes=ncpu,
                     solset='sol001',
                     observable='clock', vmin=-1., vmax=1., plot_facet_idx=True,
                     labels_in_radec=True, plot_crosses=False, phase_wrap=False,
                     flag_outliers=False)

    animate_datapack(merged_h5parm, os.path.join(working_dir, 'tec_uncert_plots'), num_processes=ncpu,
                     solset='sol001',
                     observable='weights_tec', vmin=0., vmax=10., plot_facet_idx=True,
                     labels_in_radec=True, plot_crosses=False, phase_wrap=False,
                     flag_outliers=False)

    animate_datapack(merged_h5parm, os.path.join(working_dir, 'smoothed_amp_plots'), num_processes=ncpu,
                     solset='sol001',
                     observable='amplitude', vmin=0.6, vmax=1.4, plot_facet_idx=True,
                     labels_in_radec=True, plot_crosses=False, phase_wrap=False)


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
        test_main()
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
