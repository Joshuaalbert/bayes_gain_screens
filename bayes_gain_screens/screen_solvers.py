from astropy import coordinates as ac, units as au
from jax import numpy as jnp, jit, random, vmap
from jax._src.scipy.linalg import solve_triangular
from timeit import default_timer
from bayes_gain_screens.frames import ENU
from bayes_gain_screens.tomographic_kernel import TomographicKernel
from bayes_gain_screens.utils import make_coord_array, axes_move
from bayes_gain_screens.plotting import plot_vornoi_map
from h5parm import DataPack
from jaxns import NestedSampler, plot_diagnostics, plot_cornerplot
from jaxns.gaussian_process import RBF, M32, M12, M52
from jaxns.prior_transforms import UniformPrior, PriorChain, DeltaPrior
from jaxns.utils import chunked_pmap, marginalise_static, summary
from jax.scipy.ndimage import map_coordinates
import pylab as plt


def log_normal_with_outliers(x, mean, cov, sigma):
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

def build_lookup_index(*arrays):
    def linear_lookup(values, *coords):
        fractional_coordinates = jnp.asarray([jnp.interp(coord, array, jnp.arange(array.size))
                                              for array, coord in zip(arrays, coords)])
        return map_coordinates(values, fractional_coordinates, order=1)

    return linear_lookup


def precompute_log_prob_components_with_wind(kernel, X, dtec, dtec_uncert,
                                   bottom_array, width_array, lengthscale_array, sigma_array, east_wind_speed_array,
                                   north_wind_speed_array,
                                   chunksize=2):
    """
    Precompute the log_prob for each parameter.

    Args:
        kernel:
        X:
        dtec:
        dtec_uncert:
        *arrays:

    Returns:

    """

    arrays = jnp.meshgrid(bottom_array, width_array, lengthscale_array,
                          east_wind_speed_array, north_wind_speed_array, indexing='ij')
    arrays = [a.ravel() for a in arrays]

    def compute_log_prob_components(bottom, width, lengthscale,
                                    east_wind_speed, north_wind_speed):
        wind_velocity = jnp.asarray([east_wind_speed, north_wind_speed, 0.])
        # N, N
        K = kernel(X, X, bottom, width, lengthscale, 1., wind_velocity=wind_velocity)

        def _compute_with_sigma(sigma):
            def _compute(dtec, dtec_uncert):
                return log_normal_with_outliers(dtec, 0., sigma**2 * K, dtec_uncert)

            # M
            return chunked_pmap(_compute, dtec, dtec_uncert, chunksize=1)

        # Ns,M
        return chunked_pmap(_compute_with_sigma, sigma_array, chunksize=1)

    Nb = bottom_array.shape[0]
    Nw = width_array.shape[0]
    Nl = lengthscale_array.shape[0]
    Ne = east_wind_speed_array.shape[0]
    Nn = north_wind_speed_array.shape[0]
    Ns = sigma_array.shape[0]

    # Nb*Nw*Nl*Ne*Nn,Ns,M
    log_prob = chunked_pmap(compute_log_prob_components, *arrays, chunksize=chunksize)
    # M, Nb,Nw,Nl,Ne,Nn,Ns
    log_prob = log_prob.reshape((Nb * Nw * Nl * Ne * Nn * Ns, dtec.shape[0])).transpose((1, 0)).reshape(
        (dtec.shape[0], Nb, Nw, Nl, Ne, Nn, Ns))
    return log_prob

def precompute_log_prob_components_without_wind(kernel, X, dtec, dtec_uncert,
                                   bottom_array, width_array, lengthscale_array, sigma_array,
                                   chunksize=2):
    """
    Precompute the log_prob for each parameter.

    Args:
        kernel:
        X:
        dtec:
        dtec_uncert:
        *arrays:

    Returns:

    """

    arrays = jnp.meshgrid(bottom_array, width_array, lengthscale_array, indexing='ij')
    arrays = [a.ravel() for a in arrays]

    def compute_log_prob_components(bottom, width, lengthscale):
        # N, N
        K = kernel(X, X, bottom, width, lengthscale, 1., wind_velocity=None)

        def _compute_with_sigma(sigma):
            def _compute(dtec, dtec_uncert):
                return log_normal_with_outliers(dtec, 0., sigma**2 * K, dtec_uncert)

            # M
            return chunked_pmap(_compute, dtec, dtec_uncert, chunksize=1)

        # Ns,M
        return chunked_pmap(_compute_with_sigma, sigma_array, chunksize=1)

    Nb = bottom_array.shape[0]
    Nw = width_array.shape[0]
    Nl = lengthscale_array.shape[0]
    Ns = sigma_array.shape[0]

    # Nb*Nw*Nl,Ns,M
    log_prob = chunked_pmap(compute_log_prob_components, *arrays, chunksize=chunksize)
    # M, Nb,Nw,Nl,Ns
    log_prob = log_prob.reshape((Nb * Nw * Nl * Ns, dtec.shape[0])).transpose((1, 0)).reshape(
        (dtec.shape[0], Nb, Nw, Nl, Ns))
    return log_prob

def solve_with_tomographic_kernel(dtec, dtec_uncert, X, x0, fed_kernel, time_block_size):
    """
    Precompute look-up tables for all blocks.
    Assumes that each antenna is independent and doesn't take into account time.

    Args:
        dtec: [Nd, Na, Nt]
        dtec_uncert: [Nd, Na, Nt]
        X: [Nd,6]
        x0: [3]
        fed_kernel: StationaryKernel
        time_block_size: int
    """
    scale = jnp.std(dtec) / 35.
    dtec /= scale
    dtec_uncert /= scale
    bottom_array = jnp.linspace(200., 400., 5)
    width_array = jnp.linspace(50., 50., 1)
    lengthscale_array = jnp.linspace(0.1, 7.5, 7)
    sigma_array = jnp.linspace(0.1, 2., 11)
    kernel = TomographicKernel(x0, x0, fed_kernel, S_marg=25, compute_tec=False)

    lookup_func = build_lookup_index(bottom_array, width_array, lengthscale_array, sigma_array)
    Nd,Na,Nt = dtec.shape
    remainder = Nt % time_block_size
    dtec = jnp.concatenate([dtec,dtec[:, :,-remainder:]], axis=-1)
    dtec_uncert = jnp.concatenate([dtec_uncert,dtec_uncert[:, :,-remainder:]], axis=-1)
    Nt = dtec.shape[-1]
    dtec = dtec.transpose((2,1, 0)).reshape((Nt*Na, Nd))
    dtec_uncert = dtec_uncert.transpose((2,1,0)).reshape((Nt*Na, Nd))
    # Nt*Na, ...
    log_prob = precompute_log_prob_components_without_wind(kernel, X, dtec, dtec_uncert,
                                              bottom_array, width_array, lengthscale_array, sigma_array,
                                              chunksize=4)

    log_prob = jnp.reshape(log_prob, (Nt//remainder, remainder, Na) + log_prob.shape[1:])

    def run_block(block_idx):

        def log_likelihood(bottom, width, lengthscale, sigma, **kwargs):
            return jnp.sum(vmap(lambda log_prob: lookup_func(log_prob, bottom, width, lengthscale, sigma)
                     )(log_prob[block_idx]))

        bottom = UniformPrior('bottom', bottom_array.min(), bottom_array.max())
        width = DeltaPrior('width', 50., tracked=False)
        lengthscale = UniformPrior('lengthscale', jnp.min(lengthscale_array), jnp.max(lengthscale_array))
        sigma = UniformPrior('sigma', sigma_array.min(), sigma_array.max())
        prior_chain = PriorChain(lengthscale, sigma, bottom, width)

        ns = NestedSampler(loglikelihood=log_likelihood,
                           prior_chain=prior_chain,
                           sampler_name='slice',
                           sampler_kwargs=dict(num_slices=prior_chain.U_ndims * 5),
                           num_live_points=prior_chain.U_ndims * 50)
        ns = jit(ns)
        results = ns(random.PRNGKey(42), termination_frac=0.001)

        return results
        # results.efficiency.block_until_ready()
        # t0 = default_timer()
        # results = ns(random.PRNGKey(42), termination_frac=0.001)
        # summary(results)
        # print(default_timer() - t0)

        # def screen(bottom, lengthscale, east_wind_speed, north_wind_speed, sigma, **kw):
        #     wind_velocity = jnp.asarray([east_wind_speed, north_wind_speed, 0.])
        #     K = kernel(X, X, bottom, 50., lengthscale, sigma, wind_velocity=wind_velocity)
        #     Kstar = kernel(X, Xstar, bottom, 50., lengthscale, sigma)
        #     L = jnp.linalg.cholesky(K + jnp.diag(jnp.maximum(1e-6, dtec_uncert) ** 2))
        #     dx = solve_triangular(L, dtec, lower=True)
        #     return solve_triangular(L, Kstar, lower=True).T @ dx

        # summary(results)
        # plot_diagnostics(results)
        # plot_cornerplot(results)

        # screen_mean = marginalise_static(random.PRNGKey(4325325), results.samples, results.log_p, int(results.ESS), screen)

        # print(screen_mean)
        # plot_vornoi_map(Xstar[:, 3:5], screen_mean)
        # plt.show()
        # plot_vornoi_map(X[:, 3:5], dtec)
        # plt.show()

        # return screen_mean

    results = chunked_pmap(run_block, jnp.arange(Nt//time_block_size))


def solve_with_vanilla_kernel(key, dtec, dtec_uncert, X, Xstar, fed_kernel, time_block_size, chunksize):
    """
    Precompute look-up tables for all blocks.

    Args:
        key: PRNG key
        dtec: [Nd, Na, Nt] TECU
        dtec_uncert: [Nd, Na, Nt] TECU
        X: [Nd,2] coordinates in deg
        Xstar: [Nd_screen, 2] screen coordinates
        fed_kernel: StationaryKernel
        time_block_size: int
        chunksize: int number of parallel devices to use.

    """
    field_of_view = 4. #deg
    min_separation_arcmin = 4. #drcmin
    min_separation_deg = min_separation_arcmin / 60.
    lengthscale_array = jnp.linspace(min_separation_deg, field_of_view, 120)
    sigma_array = jnp.linspace(0., 150., 150)
    kernel = fed_kernel
    lookup_func = build_lookup_index(lengthscale_array, sigma_array)

    dtec_uncert = jnp.maximum(dtec_uncert, 1e-6)

    Nd,Na,Nt = dtec.shape
    remainder = Nt % time_block_size
    extra = time_block_size - remainder
    dtec = jnp.concatenate([dtec,dtec[:, :,Nt-extra:]], axis=-1)
    dtec_uncert = jnp.concatenate([dtec_uncert,dtec_uncert[:, :,Nt-extra:]], axis=-1)
    Nt = dtec.shape[-1]
    size_dict = dict(a=Na, d=Nd, b=time_block_size)
    dtec = axes_move(dtec, ['d','a','tb'], ['atb','d'],size_dict=size_dict)
    dtec_uncert = axes_move(dtec_uncert, ['d','a','tb'], ['atb', 'd'],size_dict=size_dict)

    def compute_log_prob_components(lengthscale):
        # N, N
        K = kernel(X, X, lengthscale, 1.)
        def _compute_with_sigma(sigma):
            def _compute(dtec, dtec_uncert):
                #each [Nd]
                return log_normal_with_outliers(dtec, 0., sigma ** 2 * K, dtec_uncert)
            return chunked_pmap(_compute, dtec, dtec_uncert, chunksize=1)#M
        # Ns,M
        return chunked_pmap(_compute_with_sigma, sigma_array, chunksize=1)
    # Nl,Ns,M
    log_prob = chunked_pmap(compute_log_prob_components, lengthscale_array, chunksize=chunksize)
    # Na * (Nt//time_block_size),block_size,Nl,Ns
    log_prob = axes_move(log_prob, ['l','s','atb'],['at', 'b', 'l','s'], size_dict=size_dict)
    # Na * (Nt//time_block_size),Nl,Ns
    log_prob = jnp.sum(log_prob, axis=1)#independent datasets summed up.

    def run_block(key, dtec, dtec_uncert, log_prob):
        key1, key2 = random.split(key, 2)

        def log_likelihood(lengthscale, sigma, **kwargs):
            # K = kernel(X, X, lengthscale, sigma)
            # def _compute(dtec, dtec_uncert):
            #     #each [Nd]
            #     return log_normal_with_outliers(dtec, 0., K, jnp.maximum(1e-6, dtec_uncert))
            # return chunked_pmap(_compute, dtec, dtec_uncert, chunksize=1).sum()
            return lookup_func(log_prob, lengthscale, sigma)

        lengthscale = UniformPrior('lengthscale', jnp.min(lengthscale_array), jnp.max(lengthscale_array))
        sigma = UniformPrior('sigma', sigma_array.min(), sigma_array.max())
        prior_chain = PriorChain(lengthscale, sigma)

        ns = NestedSampler(loglikelihood=log_likelihood,
                           prior_chain=prior_chain,
                           sampler_kwargs=dict(num_slices=prior_chain.U_ndims * 1),
                           num_live_points=prior_chain.U_ndims * 50)
        ns = jit(ns)
        results = ns(key1, termination_evidence_frac=0.1)

        def marg_func(lengthscale, sigma, **kwargs):
            def screen(dtec, dtec_uncert, **kw):
                K = kernel(X, X, lengthscale, sigma)
                Kstar = kernel(X, Xstar, lengthscale, sigma)
                L = jnp.linalg.cholesky(K/(dtec_uncert[:,None]*dtec_uncert[None,:]) + jnp.eye(dtec.shape[0]))
                # L = jnp.where(jnp.isnan(L), jnp.eye(L.shape[0])/sigma, L)
                dx = solve_triangular(L, dtec/dtec_uncert, lower=True)
                JT = solve_triangular(L, Kstar/dtec_uncert[:, None], lower=True)
                #var_ik = JT_ji JT_jk
                mean = JT.T @ dx
                var = jnp.sum(JT * JT, axis=0)
                return mean, var
            return vmap(screen)(dtec, dtec_uncert), lengthscale, jnp.log(sigma)#[time_block_size,  Nd_screen], [time_block_size,  Nd_screen]

        #[time_block_size,  Nd_screen], [time_block_size,  Nd_screen], [time_block_size]
        (mean, var), mean_lengthscale, mean_logsigma = marginalise_static(key2, results.samples, results.log_p, 500, marg_func)
        uncert = jnp.sqrt(var)
        mean_sigma = jnp.exp(mean_logsigma)
        mean_lengthscale = jnp.ones(time_block_size)*mean_lengthscale
        mean_sigma = jnp.ones(time_block_size)*mean_sigma
        ESS = results.ESS*jnp.ones(time_block_size)
        logZ = results.logZ*jnp.ones(time_block_size)
        likelihood_evals = results.num_likelihood_evaluations*jnp.ones(time_block_size)
        return mean, uncert, mean_lengthscale, mean_sigma, ESS, logZ, likelihood_evals

    T = Na * (Nt//time_block_size)
    keys = random.split(key, T)
    # [T, time_block_size, Nd_screen], [T, time_block_size, Nd_screen], [T, time_block_size], [T, time_block_size]
    dtec = axes_move(dtec,['atb','d'], ['at','b','d'], size_dict=size_dict)
    dtec_uncert = axes_move(dtec_uncert,['atb','d'], ['at','b','d'], size_dict=size_dict)
    mean, uncert, mean_lengthscale, mean_sigma, ESS, logZ, likelihood_evals = chunked_pmap(run_block, keys, dtec, dtec_uncert, log_prob, chunksize=chunksize)
    mean = axes_move(mean, ['at','b','n'],['n','a','tb'], size_dict=size_dict)
    uncert = axes_move(uncert, ['at','b','n'],['n','a','tb'], size_dict=size_dict)
    mean_lengthscale = axes_move(mean_lengthscale, ['at','b'],['a','tb'], size_dict=size_dict)
    mean_sigma = axes_move(mean_sigma, ['at','b'],['a','tb'], size_dict=size_dict)
    ESS = axes_move(ESS, ['at', 'b'],['a','tb'], size_dict=size_dict)
    logZ = axes_move(logZ, ['at', 'b'],['a','tb'], size_dict=size_dict)
    likelihood_evals = axes_move(likelihood_evals, ['at', 'b'],['a','tb'], size_dict=size_dict)
    return mean[...,Nt-extra:], uncert[...,Nt-extra:], mean_lengthscale[...,Nt-extra:], mean_sigma[...,Nt-extra:], ESS[...,Nt-extra:], logZ,likelihood_evals[...,Nt-extra:]

if __name__ == '__main__':
    from jax.config import config

    config.update("jax_enable_x64", True)

    dp = DataPack('/home/albert/data/gains_screen/data/L342938_DDS5_full_merged.h5', readonly=True)
    with dp:
        select = dict(pol=slice(0, 1, 1), ant=[50], time=slice(0, 9, 1))
        dp.current_solset = 'sol000'
        dp.select(**select)
        tec_mean, axes = dp.tec
        dtec = jnp.asarray(tec_mean[0, :, :, :])
        tec_std, axes = dp.weights_tec
        dtec_uncert = jnp.asarray(tec_std[0, :, :, :])
        patch_names, directions = dp.get_directions(axes['dir'])
        antenna_labels, antennas = dp.get_antennas(axes['ant'])
        timestamps, times = dp.get_times(axes['time'])

    # antennas = ac.ITRS(*antennas.cartesian.xyz, obstime=times[0])
    # ref_ant = antennas[0]
    # frame = ENU(obstime=times[0], location=ref_ant.earth_location)
    # antennas = antennas.transform_to(frame)
    # ref_ant = antennas[0]
    # directions = directions.transform_to(frame)
    # x = antennas.cartesian.xyz.to(au.km).value.T[1:2, :]
    # k = directions.cartesian.xyz.value.T
    times = times.mjd
    times -= times[0]
    times *= 86400.

    directions = jnp.stack([directions.ra.deg, directions.dec.deg], axis=1)
    X = make_coord_array(directions, flat=True)

    n_screen = 250
    directions_star = random.uniform(random.PRNGKey(29428942), (n_screen, 2), minval=jnp.min(X, axis=0),
                           maxval=jnp.max(X, axis=0))
    Xstar = make_coord_array(directions_star, flat=True)
    #
    # kstar = random.uniform(random.PRNGKey(29428942), (n_screen, 3), minval=jnp.min(k, axis=0),
    #                        maxval=jnp.max(k, axis=0))
    # X = jnp.asarray(k)
    # Xstar = jnp.asarray(kstar)

    # print(dtec_uncert)


    mean, uncert, lengthscale, sigma, ESS, logZ, likelihood_evals = solve_with_vanilla_kernel(random.PRNGKey(42), dtec,
                                                                                              dtec_uncert, X, Xstar,
                                                                                              M32(), time_block_size=9,
                                                                                              chunksize=2)
    print(lengthscale, sigma, ESS, logZ, likelihood_evals)

    for a in range(len(antennas)):
        for t in range(len(times)):
            plot_vornoi_map(X, dtec[:,a,t])
            plt.title(f"{antenna_labels[a]}, {t}")
            plt.show()

            plot_vornoi_map(Xstar, mean[:, a, t])
            plt.title(f"{antenna_labels[a]}, {t}")
            plt.show()
            #
            # plot_vornoi_map(X, dtec_uncert[:, a, t])
            # plt.title(f"{antenna_labels[a]}, {t}")
            # plt.show()
            #
            # plot_vornoi_map(Xstar, uncert[:, a, t])
            # plt.title(f"{antenna_labels[a]}, {t}")
            # plt.show()