import matplotlib

matplotlib.use('Agg')


import argparse
import os
import numpy as np
import pylab as plt
import astropy.units as au

from jax import random, vmap, numpy as jnp, jit

from bayes_gain_screens.utils import get_screen_directions_from_image, link_overwrite, make_coord_array, axes_move, great_circle_sep
from bayes_gain_screens.screen_solvers import solve_with_vanilla_kernel
from bayes_gain_screens.plotting import make_animation, DatapackPlotter, animate_datapack

from h5parm import DataPack
from h5parm.utils import make_soltab

from jaxns.gaussian_process.kernels import RBF, M12, M32, M52

import sys
import logging

logger = logging.getLogger(__name__)

TEC_CONV = -8.4479745e6  # mTECU/Hz

@jit
def nn_interp(x,y,xstar):
    """
    Nearest-neighbour interpolation.

    Args:
        x: [N,D]
        y: [N]
        xstar: [M,D]

    Returns:
        [M] y interpolated from x to xstar.
    """
    def single_interp(xstar):
        #N
        dx = great_circle_sep(ra1=x[:,0]*jnp.pi/180.,dec1=x[:,1]*jnp.pi/180.,
                              ra2=xstar[0]*jnp.pi/180.,dec2=xstar[1]*jnp.pi/180.)
        # dx = jnp.linalg.norm(xstar - x, axis=-1)
        dx = jnp.where(dx == 0., jnp.inf, dx)
        imin = jnp.argmin(dx)
        return y[imin]
    return vmap(single_interp)(xstar)

@jit
def nn_smooth(x,y,xstar,outliers=None):
    """
    Smoothed nearest neighbours, where the smoothing length is tuned to the local nearest neighbour distance.

    Args:
        x: [N,D]
        y: [N]
        xstar: [M,D]

    Returns:
        [M] y interpolated from x to xstar.
    """
    def single_interp(xstar):
        # N
        dx = great_circle_sep(ra1=x[:, 0] * jnp.pi / 180., dec1=x[:, 1] * jnp.pi / 180.,
                              ra2=xstar[0] * jnp.pi / 180., dec2=xstar[1] * jnp.pi / 180.)
        # dx = jnp.linalg.norm(xstar - x, axis=-1)
        nn_dist = jnp.min(jnp.where(dx == 0., jnp.inf, dx))
        dx = dx/nn_dist
        weight = jnp.exp(-0.5*dx**2)
        if outliers is not None:
            weight = jnp.where(outliers, 0., weight)
        weight /= jnp.sum(weight)
        return jnp.sum(y*weight)
    return vmap(single_interp)(xstar)


def prepare_soltabs(dds5_h5parm, dds6_h5parm, screen_directions):
    logger.info("Creating sol000/phase000+amplitude000+tec000+const000")
    make_soltab(dds5_h5parm, from_solset='sol000', to_solset='sol000', from_soltab='phase000',
                to_soltab=['phase000', 'amplitude000', 'tec000', 'const000'],
                remake_solset=True,
                to_datapack=dds6_h5parm,
                directions=screen_directions)


def get_data(dds5_h5parm):
    with DataPack(dds5_h5parm, readonly=True) as h:
        select = dict(pol=slice(0, 1, 1))
        h.current_solset = 'sol000'
        h.select(**select)
        amp, axes = h.amplitude
        amp = amp[0, ...]
        phase, axes = h.phase
        phase = phase[0, ...]
        _, freqs = h.get_freqs(axes['freq'])
        freqs = freqs.to(au.Hz).value
        _, times = h.get_times(axes['time'])
        patch_names, directions = h.get_directions(axes['dir'])
        antenna_labels, antennas = h.get_antennas(axes['ant'])
        tec_mean, axes = h.tec
        tec_mean = tec_mean[0, ...]
        tec_std, axes = h.weights_tec
        tec_std = tec_std[0, ...]
        tec_outliers, _ = h.tec_outliers
        tec_outliers = tec_outliers[0,...]
        # tec_std = jnp.where(tec_outliers == 1., jnp.inf, tec_std) # already done
        const, _ = h.const
        const = const[0, ...]

    return phase, amp, tec_mean, tec_std, tec_outliers, const, antennas, directions, freqs, times

def main(data_dir, working_dir, obs_num, ref_image_fits, ncpu, max_N, plot_results):
    # os.environ['XLA_FLAGS'] = "--xla_force_host_platform_device_count={}".format(max(1,ncpu//4))

    dds5_h5parm = os.path.join(data_dir, 'L{}_DDS5_full_merged.h5'.format(obs_num))
    dds6_h5parm = os.path.join(working_dir, 'L{}_DDS6_full_merged.h5'.format(obs_num))
    linked_dds6_h5parm = os.path.join(data_dir, 'L{}_DDS6_full_merged.h5'.format(obs_num))
    logger.info("Looking for {}".format(dds5_h5parm))
    link_overwrite(dds6_h5parm, linked_dds6_h5parm)

    phase, amp, dtec_mean, dtec_std, tec_outliers, const, antennas, directions, freqs, times = get_data(dds5_h5parm)
    Nd, Na, Nf, Nt = amp.shape

    times = times.mjd
    times -= times[0]
    times *= 86400.
    dt = jnp.mean(jnp.diff(times))

    #ICRS
    screen_directions, _ = get_screen_directions_from_image(ref_image_fits,
                                                            flux_limit=0.01,
                                                            max_N=max_N,
                                                            min_spacing_arcmin=4.,
                                                            seed_directions=directions,
                                                            fill_in_distance=8.,
                                                            fill_in_flux_limit=0.01)

    prepare_soltabs(dds5_h5parm, dds6_h5parm, screen_directions)

    directions = jnp.stack([directions.ra.deg, directions.dec.deg], axis=1)
    X = make_coord_array(directions, flat=True)
    screen_directions = jnp.stack([screen_directions.ra.deg, screen_directions.dec.deg], axis=1)
    Xstar = make_coord_array(screen_directions, flat=True)

    regularity_window = 5.#minutes over which ionosphere properties remain the same
    time_block_size = max(1,int(regularity_window*60./dt))
    logger.info(f"Ionosphere properties assumed constant over {regularity_window} minutes ({time_block_size} timesteps).")

    post_dtec_screen_mean, post_dtec_screen_uncert, lengthscale, sigma, ESS, logZ, likelihood_evals = \
        solve_with_vanilla_kernel(random.PRNGKey(42),
                                  dtec=dtec_mean, dtec_uncert=dtec_std,
                              X=X, Xstar=Xstar,fed_kernel=M32(),
                              time_block_size=time_block_size,#assume screen hyper-parameters are constant over 10 time-steps.
                              chunksize=max(1,ncpu//4))

    interp_type = 'nearest_neighbour'
    amp = axes_move(amp, ['d','a','f','t'],['aft','d'])
    const = axes_move(const,['d','a','t'],['at','d'])
    if interp_type == 'nearest_neighbour':
        post_amp = vmap(lambda amp: nn_interp(directions,amp,screen_directions))(amp.T).T
        post_const = nn_interp(directions, const, screen_directions)
    elif interp_type == 'smoothed_nearest_neighbour':
        post_amp = vmap(lambda amp: nn_smooth(directions, amp, screen_directions))(amp.T).T
        post_const = nn_smooth(directions, const, screen_directions)
    else:
        raise ValueError(f"Invalid interp_type {interp_type}")
    post_amp = axes_move(post_amp, ['aft', 'd'], ['d', 'a', 'f', 't'],size_dict=dict(a=Na,f=Nf,t=Nt))
    post_const = axes_move(post_const, ['at', 'd'], ['d', 'a', 't'],size_dict=dict(a=Na,t=Nt))

    post_phase = post_dtec_screen_mean[...,None,:]*(TEC_CONV/freqs[:,None]) + post_const[..., None,:]
    post_uncert = jnp.abs(post_dtec_screen_uncert[...,None,:]*(TEC_CONV/freqs[:,None]))

    #Replace outliers with screen solutions, else the calibrators.
    phase_outliers_replaced_with_screen = jnp.where(tec_outliers[...,None,:], post_phase[:Nd], phase)

    with DataPack(dds6_h5parm, readonly=False) as h:
        h.current_solset = 'sol000'
        # set screen
        h.select(pol=slice(0, 1, 1))
        h.tec = np.asarray(post_dtec_screen_mean)[None, ...]
        h.weights_tec = np.asarray(post_dtec_screen_uncert)[None, ...]
        h.const = np.asarray(post_const)[None, ...]
        h.amplitude = np.asarray(post_amp)[None, ...]
        h.phase = np.asarray(post_phase)[None, ...]
        h.weights_phase = np.asarray(post_uncert)[None, ...]
        # put calibrators in original diretions
        h.select(pol=slice(0, 1, 1), dir=slice(0, Nd, 1))
        h.phase = np.asarray(phase_outliers_replaced_with_screen)[None, ...]
        h.amplitude = np.asarray(amp)[None, ...]

    # replace the outlier phase calibrators in dds5
    with DataPack(dds5_h5parm, readonly=False) as h:
        h.current_solset = 'sol000'
        h.select(pol=slice(0, 1, 1))
        h.phase = np.asarray(phase_outliers_replaced_with_screen)[None, ...]

    if plot_results:
        logger.info("Plotting results.")
        d = os.path.join(working_dir, 'tec_screen_plots')
        animate_datapack(dds6_h5parm, d, ncpu,
                         vmin=-60,
                         vmax=60., observable='tec', phase_wrap=False, plot_crosses=False,
                         plot_facet_idx=False, labels_in_radec=True, per_timestep_scale=True,
                         solset='sol000', cmap=plt.cm.PuOr
                         )
        d = os.path.join(working_dir, 'const_screen_plots')
        animate_datapack(dds6_h5parm, d, ncpu,
                         vmin=-np.pi,
                         vmax=np.pi, observable='const', phase_wrap=True, plot_crosses=False,
                         plot_facet_idx=False, labels_in_radec=True, per_timestep_scale=False,
                         solset='sol000')
        d = os.path.join(working_dir, 'amp_screen_plots')
        animate_datapack(dds6_h5parm, d, ncpu,
                         vmin=1.5,
                         vmax=0.5, observable='amp', phase_wrap=False, plot_crosses=False,
                         plot_facet_idx=False, labels_in_radec=True, per_timestep_scale=True,
                         solset='sol000', cmap=plt.cm.PuOr
                         )
        # os.makedirs(d, exist_ok=True)
        # DatapackPlotter(dds6_h5parm).plot(
        #     fignames=[os.path.join(d, "fig-{:04d}.png".format(j)) for j in range(Nt)],
        #     vmin=-60,
        #     vmax=60., observable='tec', phase_wrap=False, plot_crosses=False,
        #     plot_facet_idx=False, labels_in_radec=True, per_timestep_scale=True,
        #     solset='sol000', cmap=plt.cm.PuOr)
        # make_animation(d, prefix='fig', fps=4)

        # d = os.path.join(working_dir, 'const_screen_plots')
        # os.makedirs(d, exist_ok=True)
        # DatapackPlotter(dds6_h5parm).plot(
        #     fignames=[os.path.join(d, "fig-{:04d}.png".format(j)) for j in range(Nt)],
        #     vmin=-np.pi,
        #     vmax=np.pi, observable='const', phase_wrap=False, plot_crosses=False,
        #     plot_facet_idx=False, labels_in_radec=True, per_timestep_scale=True,
        #     solset='sol000', cmap=plt.cm.PuOr)
        # make_animation(d, prefix='fig', fps=4)
        #
        # d = os.path.join(working_dir, 'clock_screen_plots')
        # os.makedirs(d, exist_ok=True)
        # DatapackPlotter(dds6_h5parm).plot(
        #     fignames=[os.path.join(d, "fig-{:04d}.png".format(j)) for j in range(Nt)],
        #     vmin=None,
        #     vmax=None,
        #     observable='clock', phase_wrap=False, plot_crosses=False,
        #     plot_facet_idx=False, labels_in_radec=True, per_timestep_scale=True,
        #     solset='sol000', cmap=plt.cm.PuOr)
        # make_animation(d, prefix='fig', fps=4)
        #
        # d = os.path.join(working_dir, 'amplitude_screen_plots')
        # os.makedirs(d, exist_ok=True)
        # DatapackPlotter(dds6_h5parm).plot(
        #     fignames=[os.path.join(d, "fig-{:04d}.png".format(j)) for j in range(Nt)],
        #     log_scale=True, observable='amplitude', phase_wrap=False, plot_crosses=False,
        #     plot_facet_idx=False, labels_in_radec=True, per_timestep_scale=True,
        #     solset='sol000', cmap=plt.cm.PuOr)
        # make_animation(d, prefix='fig', fps=4)


def debug_main():
    main(data_dir='/home/albert/data/gains_screen/data',
         working_dir='/home/albert/data/gains_screen/working_dir/',
         obs_num=342938,
         ref_image_fits='/home/albert/data/gains_screen/data/lotss_archive_deep_image.app.restored.fits',
         ncpu=1,
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
                        default=True, type="bool", required=False)


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
