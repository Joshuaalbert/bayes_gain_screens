import argparse
import os
from timeit import default_timer

from jax import random, vmap, numpy as jnp
from jax.scipy.linalg import solve_triangular
from jax.scipy.special import erf

from bayes_gain_screens.utils import chunked_pmap, get_screen_directions_from_image
from bayes_gain_screens.plotting import animate_datapack

from h5parm import DataPack
from h5parm.utils import make_soltab

from jaxns.gaussian_process.kernels import RBF
from jaxns.prior_transforms import UniformPrior, PriorChain, HalfLaplacePrior, GaussianProcessKernelPrior
from jaxns.nested_sampling import NestedSampler

import logging

logger = logging.getLogger(__name__)



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


def screen_model(key, tec_mean, tec_std, const, clock, directions, outliers):
    def single_screen(key, tec_mean, tec_std, outliers):
        kernel = RBF()
        l = UniformPrior('l', 0., 10. * jnp.pi / 180.)
        sigma = UniformPrior('sigma', 0., 2. * jnp.std(tec_mean))
        K = GaussianProcessKernelPrior('K', kernel, directions,
                                       l,
                                       sigma,
                                       tracked=False)
        uncert = jnp.where(outliers, jnp.inf, tec_std)

        def log_likelihood(K, **kwargs):
            return log_normal_outliers(tec_mean, 0., K, uncert)

        prior_chain = PriorChain().push(K)

        def predict_f(K, **kwargs):
            # (K + sigma.sigma)^-1 = sigma^-1.(sigma^-1.K.sigma^-1 + I)^-1.sigma^-1
            C = K / (uncert[:, None] * uncert[None, :]) + jnp.eye(K.shape[0])
            JT = jnp.linalg.solve(C, K / uncert[:, None])
            return JT.T @ (tec_mean / uncert)

        def predict_fvar(K, **kwargs):
            C = K / (uncert[:, None] * uncert[None, :]) + jnp.eye(K.shape[0])
            JT = jnp.linalg.solve(C, K / uncert[:, None])
            return jnp.diag(K - JT.T @ (K / uncert[:, None]))

        ns = NestedSampler(log_likelihood,
                           prior_chain,
                           sampler_name='slice',
                           predict_tec_mean=predict_f,
                           predict_tec_var=predict_fvar
                           )

        results = ns(key=key,
                     num_live_points=500,
                     max_samples=1e6,
                     collect_samples=False,
                     termination_frac=0.001,
                     sampler_kwargs=dict(depth=2, num_slices=5))
        post_tec_mean = results.marginalised['predict_tec_mean']
        post_tec_std = jnp.sqrt(results.marginalised['predict_tec_var'])
        return post_tec_mean, post_tec_std

    Nt, Na, Nd = tec_mean.shape
    tec_mean = tec_mean.reshape((Nt * Na, Nd))
    tec_std = tec_std.reshape((Nt * Na, Nd))
    outliers = outliers.reshape((Nt * Na, Nd))
    T = Nt * Na
    post_tec_mean, post_tec_std = chunked_pmap(single_screen, random.split(key, T), tec_mean, tec_std, outliers,
                                               chunksize=2)
    return post_tec_mean.reshape((Nt, Na, Nd)), post_tec_std.reshape((Nt, Na, Nd))


def drop_array(n, m):
    # TODO to with mod n
    a = jnp.arange(n)
    a = jnp.roll(a, -m, axis=0)
    a = a[1:]
    a = jnp.roll(a, m, axis=0)
    return a


def inverse_update(C, m, return_drop=False):
    drop = drop_array(C.shape[0], m)
    _a = jnp.take(C, drop, axis=0)  # drop m row
    a = jnp.take(_a, drop, axis=1)
    c = jnp.take(C, drop, axis=1)[None, m, :]  # drop m col
    b = _a[:, m, None]
    d = C[m, m]
    res = a - (b @ c) / d
    if return_drop:
        return res, drop
    return res


def outlier_prob(uncert, K, Y_obs, kappa=2.5):
    Sigma = jnp.diag(uncert ** 2)  # * jnp.eye(K.shape[0])
    C = K + Sigma
    Cinv = jnp.linalg.pinv(C)

    def single_log_likelihood(m):
        Cinv_reduced, drop = inverse_update(Cinv, m, return_drop=True)
        kstar_reduced = jnp.take(K[m, :], drop, axis=0)
        JT = Cinv_reduced @ kstar_reduced
        sigma2_star = K[m, m] - kstar_reduced @ JT + uncert[m] ** 2
        Y_obs_reduced = jnp.take(Y_obs, drop, axis=0)
        mu_star = JT @ Y_obs_reduced
        z = jnp.maximum(jnp.abs(Y_obs[m] - mu_star) / jnp.sqrt(sigma2_star), kappa)
        return (erf(-kappa / jnp.sqrt(2)) - erf(-z / jnp.sqrt(2))) / (1. + erf(-kappa / jnp.sqrt(2)))

    return vmap(single_log_likelihood)(jnp.arange(K.shape[0]))


def detect_outliers(key, tec_mean, tec_std, directions, kappa=2.5):
    def single_detect_outliers(key, tec_mean, tec_std):
        kernel = RBF()
        prob_outlier = outlier_prob(tec_std, kernel(directions, directions, 0.6 * jnp.pi / 180., jnp.std(tec_mean)),
                                    tec_mean, kappa=kappa)
        outliers = prob_outlier > 0.75

        return prob_outlier, outliers

    Nt, Na, Nd = tec_mean.shape
    tec_mean = tec_mean.reshape((Nt * Na, Nd))
    tec_std = tec_std.reshape((Nt * Na, Nd))
    T = Nt * Na
    prob_outliers, outliers = chunked_pmap(single_detect_outliers, random.split(key, T), tec_mean, tec_std)
    return prob_outliers.reshape(Nt, Na, Nd), outliers.reshape(Nt, Na, Nd)


def link_overwrite(src, dst):
    if os.path.islink(dst):
        print("Unlinking pre-existing sym link {}".format(dst))
        os.unlink(dst)
    print("Linking {} -> {}".format(src, dst))
    os.symlink(src, dst)


def prepare_soltabs(dds5_h5parm, dds6_h5parm, screen_directions):
    logger.info("Creating sol000/phase000+amplitude000+tec000+const000+clock000")
    make_soltab(dds5_h5parm, from_solset='sol000', to_solset='sol000', from_soltab='phase000',
                to_soltab=['phase000', 'amplitude000', 'tec000', 'const000', 'clock000'],
                remake_solset=True,
                to_datapack=dds6_h5parm,
                directions=screen_directions)


def generate_data(dds5_h5parm):
    with DataPack(dds5_h5parm, readonly=True) as h:
        select = dict(pol=slice(0, 1, 1))
        h.current_solset = 'sol000'
        h.select(**select)
        tec_mean, axes = h.tec
        tec_std, axes = h.weights_tec
        const, _ = h.const
        clock, _ = h.clock
        patch_names, directions = h.get_directions(axes['dir'])
        directions = jnp.stack([directions.ra.rad, directions.dec.rad], axis=-1)
    return tec_mean, tec_std, const, clock, directions


def main(data_dir, working_dir, obs_num, ref_image_fits, ncpu, max_N):
    dds5_h5parm = os.path.join(working_dir, 'L{}_DDS5_full_merged.h5'.format(obs_num))
    dds6_h5parm = os.path.join(working_dir, 'L{}_DDS6_full_merged.h5'.format(obs_num))
    linked_dds6_h5parm = os.path.join(data_dir, 'L{}_DDS6_full_merged.h5'.format(obs_num))
    logger.info("Looking for {}".format(dds5_h5parm))
    link_overwrite(dds6_h5parm, linked_dds6_h5parm)

    tec_mean, tec_std, const, clock, directions = generate_data(dds5_h5parm)

    screen_directions, _ = get_screen_directions_from_image(ref_image_fits, flux_limit=0.01, max_N=max_N,
                                                            min_spacing_arcmin=4.,
                                                            seed_directions=directions,
                                                            fill_in_distance=8.,
                                                            fill_in_flux_limit=0.01)

    prepare_soltabs(dds5_h5parm, dds6_h5parm, screen_directions)


    _, outliers = detect_outliers(random.PRNGKey(int(default_timer())), tec_mean, tec_std, directions, kappa=2.5)

    post_tec_mean, post_const, post_clock = screen_model(random.PRNGKey(int(default_timer())), tec_mean, tec_std, const, clock, directions, outliers)

    with DataPack(dds6_h5parm, readonly=False) as h:
        h.current_solset = 'sol000'
        h.select(pol=slice(0,1,1))
        h.tec = post_tec_mean
        h.const = post_const
        h.clock = post_clock

    animate_datapack(dds6_h5parm, os.path.join(working_dir, 'tec_screen_plots'), num_processes=ncpu,
                     solset='sol000',
                     observable='tec', vmin=-60., vmax=60., labels_in_radec=True, plot_crosses=False, phase_wrap=False)

    animate_datapack(dds6_h5parm, os.path.join(working_dir, 'const_screen_plots'), num_processes=ncpu,
                     solset='sol000',
                     observable='const', vmin=-jnp.pi, vmax=jnp.pi, labels_in_radec=True, plot_crosses=False, phase_wrap=True)

    animate_datapack(dds6_h5parm, os.path.join(working_dir, 'clock_screen_plots'), num_processes=ncpu,
                     solset='sol000',
                     observable='clock', vmin=-1., vmax=1., labels_in_radec=True, plot_crosses=False, phase_wrap=False)


def test_deployment():
    main(data_dir='/home/albert/store/lockman/test/root/L667218/subtract',
         working_dir='/home/albert/store/lockman/test/root/L667218/infer_screen',
         obs_num=667218,
         ref_dir=0,
         deployment_type='directional',
         block_size=10,
         ref_image_fits='/home/albert/store/lockman/lotss_archive_deep_image.app.restored.fits')


def add_args(parser):
    parser.add_argument('--obs_num', help='Obs number L*',
                        default=None, type=int, required=True)
    parser.add_argument('--data_dir', help='Where are the ms files are stored.',
                        default=None, type=str, required=True)
    parser.add_argument('--working_dir', help='Where to perform the imaging.',
                        default=None, type=str, required=True)
    parser.add_argument('--ref_dir', help='The index of reference dir.',
                        default=0, type=int, required=False)
    parser.add_argument('--ncpu', help='Number of CPUs.',
                        default=32, type=int, required=True)
    parser.add_argument('--deployment_type', help='The type of screen [directional, non_integral, tomographic].',
                        default='directional', type=str, required=False)
    parser.add_argument('--block_size', help='The number of time steps to process at once.',
                        default=10, type=int, required=False)
    parser.add_argument('--max_N', help='The maximum number of screen directions.',
                        default=250, type=int, required=False)
    parser.add_argument('--ref_image_fits',
                        help='The Gaussian source list of the field used to choose locations of screen points.',
                        type=str, required=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Infers the value of DDTEC and a constant over a screen.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_args(parser)
    flags, unparsed = parser.parse_known_args()
    print("Running with:")
    for option, value in vars(flags).items():
        print("\t{} -> {}".format(option, value))
    main(**vars(flags))
