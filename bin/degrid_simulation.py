import argparse
import sys
import logging
logger = logging.getLogger(__name__)


from bayes_gain_screens.tomographic_kernel import TomographicKernel, GeodesicTuple
from bayes_gain_screens.tomographic_kernel.tomographic_kernel import scan_vmap
from bayes_gain_screens.utils import make_coord_array, wrap, great_circle_sep
from bayes_gain_screens.plotting import plot_vornoi_map
from bayes_gain_screens.frames import ENU
from h5parm import DataPack
import os
from jaxns.gaussian_process.kernels import RBF, M32
from jaxns.utils import chunked_pmap
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
from jax import jit, random, vmap, tree_map
from h5parm.utils import create_empty_datapack
from h5parm import DataPack
import astropy.units as au
import astropy.coordinates as ac
import astropy.time as at
import pylab as plt
import numpy as np
from tqdm import tqdm
from bayes_gain_screens.utils import chunked_pmap
from timeit import default_timer

ARRAYS = {'lofar': DataPack.lofar_array_hba,
          'dsa2000W': './dsa2000.W.cfg',
          'dsa2000W10': './dsa2000.W.10.cfg',
          'dsa2000W_200m_grid': './dsa2000.W.200m_grid.cfg',
          'dsa2000W_300m_grid': './dsa2000.W.300m_grid.cfg',
          'dsa2000W_400m_grid': './dsa2000.W.400m_grid.cfg',
          'dsa2000W_500m_grid': './dsa2000.W.500m_grid.cfg',
          'dsa2000W_600m_grid': './dsa2000.W.600m_grid.cfg',
          'dsa2000W_700m_grid': './dsa2000.W.700m_grid.cfg',
          'dsa2000W_800m_grid': './dsa2000.W.800m_grid.cfg',
          'dsa2000W_900m_grid': './dsa2000.W.900m_grid.cfg',
          'dsa2000W_1000m_grid': './dsa2000.W.1000m_grid.cfg',
          }

def interpolation(dtec_in, antennas_in: ac.ITRS, directions_in: ac.ICRS, times_in: at.Time,
                  antennas_out: ac.ITRS, directions_out: ac.ICRS, times_out: at.Time,
                  k=3):
    """
    Interpolate dtec_in onto out coordinates using linear interpolation.

    Args:
        dtec_in: [Nd, Na, Nt]
        antennas_in: [Na]
        directions_in: [Nd]
        times_in: [Nt]
        antennas_out: [Na']
        directions_out: [Nd']
        times_out: [Nt']

    Returns:
        dtec_out [Nd', Na', Nt']
    """
    logger.info("Interpolating...")
    antennas_in = antennas_in.cartesian.xyz.to(au.km).value.T
    antennas_out = antennas_out.cartesian.xyz.to(au.km).value.T
    directions_in = np.stack([directions_in.ra.rad, directions_in.dec.rad], axis=1)
    directions_out = np.stack([directions_out.ra.rad, directions_out.dec.rad], axis=1)
    times_in = times_in.mjd*86400.
    times_in -= times_in[0]
    times_out = times_out.mjd * 86400.
    times_out -= times_out[0]

    from scipy.spatial.distance import cdist
    antennas_dist = cdist(antennas_in, antennas_out, metric='euclidean')
    directions_dist = cdist(directions_in, directions_out, metric=lambda k1, k2: great_circle_sep(*k1, *k2))
    times_dist = cdist(times_in, times_out, metric='euclidean')
    # antennas_first
    dtec_out = np.zeros((directions_in.shape[0], antennas_out.shape[0], times_in.shape[0]))
    closest_ants = np.argsort(antennas_dist, axis=0)
    for i, x_out in enumerate(antennas_out):
        dists = antennas_dist[closest_ants[:k,i], i]
        print(dists)
        kernel = np.exp(-0.5*(dists/1.)**2)
        kernel /= kernel.sum()
        dtec_out[:, i, :] = np.sum(dtec_in[:,closest_ants[:k,i],:] * kernel[None,:, None], axis=1)

    # directions_next
    dtec_in = dtec_out
    dtec_out = np.zeros((directions_out.shape[0], antennas_out.shape[0], times_in.shape[0]))
    closest_dirs = np.argsort(directions_dist, axis=0)
    for i, x_out in enumerate(directions_out):
        dists = directions_dist[closest_dirs[:k, i], i]
        print(dists)
        kernel = np.exp(-0.5 * (dists / 1.) ** 2)
        kernel /= kernel.sum()
        dtec_out[i, :, :] = np.sum(dtec_in[closest_dirs[:k, i], :, :] * kernel[:, None, None], axis=0)

    # times last
    dtec_in = dtec_out
    dtec_out = np.zeros((directions_out.shape[0], antennas_out.shape[0], times_out.shape[0]))
    closest_times = np.argsort(times_dist, axis=0)
    for i, x_out in enumerate(times_out):
        dists = directions_dist[closest_times[:k, i], i]
        print(dists)
        kernel = np.exp(-0.5 * (dists / 1.) ** 2)
        kernel /= kernel.sum()
        dtec_out[:, :, i] = np.sum(dtec_in[:, :, closest_times[:k, i]] * kernel[None, None, :], axis=2)

    return dtec_out




def create_empty_datapack_spec(directions: ac.ICRS,
                               Nf,
                               Nt,
                               pols=None,
                              start_time=None,
                              time_resolution=30.,
                              min_freq=122.,
                              max_freq=166.,
                              array_file=None,
                              save_name='test_datapack.h5',
                              clobber=False) -> DataPack:
    """
    Creates an empty datapack with phase, amplitude and DTEC.

    Args:
        Nd: number of directions
        Nf: number of frequencies
        Nt: number of times
        pols: polarisations, ['XX', ...]
        array_file: array file else Lofar HBA is used
        phase_tracking: tuple (RA, DEC) in degrees in ICRS frame
        field_of_view_diameter: FoV diameter in degrees
        start_time: start time in modified Julian days (mjs/86400)
        time_resolution: time step in seconds.
        min_freq: minimum frequency in MHz
        max_freq: maximum frequency in MHz
        save_name: where to save the H5parm.
        clobber: Whether to overwrite.

    Returns:
        DataPack
    """

    logger.info("=== Creating empty datapack ===")
    Nd = len(directions)

    save_name = os.path.abspath(save_name)
    if os.path.isfile(save_name) and clobber:
        os.unlink(save_name)

    if array_file is None:
        array_file = DataPack.lofar_array_hba

    if start_time is None:
        start_time = at.Time("2019-01-01T00:00:00.000", format='isot').mjd

    if pols is None:
        pols = ['XX']
    assert isinstance(pols, (tuple, list))

    time0 = at.Time(start_time, format='mjd')

    datapack = DataPack(save_name, readonly=False)
    with datapack:
        datapack.add_solset('sol000', array_file=array_file)
        datapack.set_directions(None, np.stack([directions.ra.rad, directions.dec.rad], axis=1))

        patch_names, _ = datapack.directions
        antenna_labels, _ = datapack.antennas
        _, antennas = datapack.get_antennas(antenna_labels)
        antennas = antennas.cartesian.xyz.to(au.km).value.T
        Na = antennas.shape[0]

        times = at.Time(time0.mjd + (np.arange(Nt) * time_resolution)/86400., format='mjd').mjd * 86400.  # mjs
        freqs = np.linspace(min_freq, max_freq, Nf) * 1e6


        Npol = len(pols)
        dtecs = np.zeros((Npol, Nd, Na, Nt))
        phase = np.zeros((Npol, Nd, Na, Nf, Nt))
        amp = np.ones_like(phase)

        datapack.add_soltab('phase000', values=phase, ant=antenna_labels, dir=patch_names, time=times, freq=freqs,
                            pol=pols)
        datapack.add_soltab('amplitude000', values=amp, ant=antenna_labels, dir=patch_names, time=times, freq=freqs,
                            pol=pols)
        datapack.add_soltab('tec000', values=dtecs, ant=antenna_labels, dir=patch_names, time=times, pol=pols)
        return datapack

class Degrid(object):
    def __init__(self, input_datapack, output_datapack, array_name):
        self._input_datapack = input_datapack
        self._output_datapack = output_datapack
        self._array_file = ARRAYS[array_name]

    def run(self, duration, time_resolution, start_time):

        with DataPack(self._input_datapack, readonly=True) as dp:
            dp.current_solset = 'sol000'
            dp.select(pol=slice(0, 1, 1))
            tec_grid, axes = dp.tec
            patch_names, directions_grid = dp.get_directions(axes['dir'])
            antenna_labels, antennas_grid = dp.get_antennas(axes['ant'])
            timestamps, times_grid = dp.get_times(axes['time'])

        Nf = 2  # 8000
        Nt = int(duration / time_resolution) + 1
        min_freq = 700.
        max_freq = 2000.

        dp_out = create_empty_datapack_spec(directions_grid, Nf, Nt,pols=None,
                                   start_time=start_time,time_resolution=time_resolution,
                                   min_freq=min_freq,
                                   max_freq=max_freq, array_file=self._array_file,
                                   save_name=self._output_datapack,
                                   clobber=False)

        logger.info("Loading gridded data...")
        with dp_out:
            dp.current_solset = 'sol000'
            dp.select(pol=slice(0, 1, 1))
            patch_names, directions_out = dp.get_directions(axes['dir'])
            antenna_labels, antennas_out = dp.get_antennas(axes['ant'])
            timestamps, times_out = dp.get_times(axes['time'])

        dtec_out = interpolation(tec_grid, antennas_grid, directions_grid, times_grid,
                      antennas_out, directions_out, times_out,k=4)

        logger.info("Saving interpolated H5parm...")
        with dp_out:
            dp.current_solset = 'sol000'
            dp.select(pol=slice(0, 1, 1))
            dp.tec = dtec_out

def main(input_h5parm, output_h5parm,
         array_name, start_time, time_resolution, duration):

    Degrid(input_h5parm, output_h5parm, array_name).run(duration,time_resolution, start_time)


def debug_main():

    main(input_h5parm="",
         output_h5parm='test_dsa2000W_datapack.h5',
         array_name='dsa2000W_1000m_grid',
         start_time=at.Time('2019-03-19T19:58:14.9', format='isot'),
         time_resolution=60.,
         duration=0.)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        debug_main()
        exit(0)