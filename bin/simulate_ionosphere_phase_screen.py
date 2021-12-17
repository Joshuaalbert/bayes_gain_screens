import argparse
import sys
import logging
logger = logging.getLogger(__name__)

from jax.config import config
config.update("jax_enable_x64", True)

from bayes_gain_screens.tomographic_kernel import TomographicKernel, GeodesicTuple
from bayes_gain_screens.utils import make_coord_array, wrap
from bayes_gain_screens.plotting import plot_vornoi_map
from bayes_gain_screens.frames import ENU
from h5parm import DataPack
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
          'dsa2000W_1500m_grid': './dsa2000.W.1500m_grid.cfg',
          'dsa2000W_2000m_grid': './dsa2000.W.2000m_grid.cfg',
          }

def get_num_directions(avg_spacing, field_of_view_diameter, min_n=1):
    """
    Get the number of directions that will space the field of view by the given spacing.

    Args:
        avg_spacing:
        field_of_view_diameter:

    Returns:
        int, the number of directions to sample inside the S^2
    """
    V = 2.*np.pi*(field_of_view_diameter/2.)**2
    pp = 0.5
    n = -V * np.log(1. - pp) / (avg_spacing/60.)**2 / np.pi / 2.
    n = max(int(n), min_n)
    return n

def visualisation(h5parm, ant=None, time=None):
    with DataPack(h5parm, readonly=True) as dp:
        dp.current_solset = 'sol000'
        dp.select(ant=ant, time=time)
        dtec, axes = dp.tec
        dtec = dtec[0]
        patch_names, directions = dp.get_directions(axes['dir'])
        antenna_labels, antennas = dp.get_antennas(axes['ant'])
        timestamps, times = dp.get_times(axes['time'])

    frame = ENU(obstime=time, location=antennas[0].earth_location)
    directions = directions.transform_to(frame)
    t = times.mjd * 86400.
    t -= t[0]
    dt = np.diff(t).mean()
    x = antennas.cartesian.xyz.to(au.km).value.T[1:, :]
    # x[1,:] = x[0,:]
    # x[1,0] += 0.3
    k = directions.cartesian.xyz.value.T
    logger.info(f"Directions: {directions}")
    logger.info(f"Antennas: {x} {antenna_labels}")
    logger.info(f"Times: {t}")
    Na = x.shape[0]
    logger.info(f"Number of antenna to plot: {Na}")
    Nd = k.shape[0]
    Nt = t.shape[0]


    fig, axs = plt.subplots(Na, Nt, sharex=True, sharey=True,
                            figsize=(2 * Nt, 2 * Na),
                            squeeze=False)

    for a in range(Na):
        for i in range(Nt):
            ax = axs[a][i]
            ax = plot_vornoi_map(k[:, 0:2], dtec[:, a, i], ax=ax, colorbar=False)
            if a == (Na - 1):
                ax.set_xlabel(r"$k_{\rm east}$")
            if i == 0:
                ax.set_ylabel(r"$k_{\rm north}$")
            if a == 0:
                ax.set_title(f"Time: {int(t[i])} sec")

    plt.show()

class Simulation(object):
    def __init__(self, wind_vector,bottom,width,l,fed_mu,fed_sigma):
        """
        Simulation of DTEC.

        Args:
            wind_vector: Tangential velocity at bottom in km/s
            bottom: bottom of ionosphere layer in km
            width: thickness of the ionosphere layer in km
            l: lengthscale of FED irregularities in km
            fed_mu: FED mean density in mTECU / km = 10^10 e/m^3
            fed_sigma: FED variation of spatial Gaussian process in mTECU / km = 10^10 e/m^3
        """
        self._wind_vector = wind_vector
        self._bottom = bottom
        self._width = width
        self._l = l
        self._fed_kernel_params = dict(l=l, sigma = 1.)
        self._fed_mu = fed_mu
        self._fed_sigma = fed_sigma
        logger.info(f"Simulation parameters:\n"
                    f"bottom={bottom} km\n"
                    f"width={width} km\n"
                    f"lengthscale={l} km\n"
                    f"fed_mu={fed_mu} mTECU/km\n"
                    f"fed_sigma={fed_sigma} mTECU/km")

    def run(self, output_h5parm, ncpu, avg_direction_spacing, field_of_view_diameter, duration, time_resolution, start_time, array_name,
                   phase_tracking):

        Nd = get_num_directions(avg_direction_spacing, field_of_view_diameter, )
        Nf = 2  # 8000
        Nt = int(duration / time_resolution) + 1
        min_freq = 700.
        max_freq = 2000.
        dp = create_empty_datapack(Nd, Nf, Nt, pols=None,
                                   field_of_view_diameter=field_of_view_diameter,
                                   start_time=start_time,
                                   time_resolution=time_resolution,
                                   min_freq=min_freq,
                                   max_freq=max_freq,
                                   array_file=ARRAYS[array_name],
                                   phase_tracking=(phase_tracking.ra.deg, phase_tracking.dec.deg),
                                   save_name=output_h5parm,
                                   clobber=True)

        with dp:
            dp.current_solset = 'sol000'
            dp.select(pol=slice(0, 1, 1))
            axes = dp.axes_tec
            patch_names, directions = dp.get_directions(axes['dir'])
            antenna_labels, antennas = dp.get_antennas(axes['ant'])
            timestamps, times = dp.get_times(axes['time'])
            ref_ant = antennas[0]
            ref_time = times[0]

        Na = len(antennas)
        Nd = len(directions)
        Nt = len(times)

        logger.info(f"Number of directions: {Nd}")
        logger.info(f"Number of antennas: {Na}")
        logger.info(f"Number of times: {Nt}")
        logger.info(f"Reference Ant: {ref_ant}")
        logger.info(f"Reference Time: {ref_time.isot}")


        # Plot Antenna Layout in East North Up frame
        ref_frame = ENU(obstime=ref_time, location=ref_ant.earth_location)

        _antennas = ac.ITRS(*antennas.cartesian.xyz, obstime=ref_time).transform_to(ref_frame)
        # plt.scatter(_antennas.east, _antennas.north, marker='+')
        # plt.xlabel(f"East (m)")
        # plt.ylabel(f"North (m)")
        # plt.show()

        x0 = ac.ITRS(*antennas[0].cartesian.xyz, obstime=ref_time).transform_to(ref_frame).cartesian.xyz.to(au.km).value
        earth_centre_x = ac.ITRS(x=0 * au.m, y=0 * au.m, z=0. * au.m, obstime=ref_time).transform_to(
            ref_frame).cartesian.xyz.to(au.km).value
        self._kernel = TomographicKernel(x0, earth_centre_x, M32(), S_marg=20, compute_tec=False)

        k = directions.transform_to(ref_frame).cartesian.xyz.value.T

        t = times.mjd * 86400.
        t -= t[0]

        X1 = GeodesicTuple(x=[], k=[], t=[], ref_x=[])

        logger.info("Computing coordinates in frame ...")

        for i, time in tqdm(enumerate(times)):
            x = ac.ITRS(*antennas.cartesian.xyz, obstime=time).transform_to(ref_frame).cartesian.xyz.to(
                au.km).value.T
            ref_ant_x = ac.ITRS(*ref_ant.cartesian.xyz, obstime=time).transform_to(ref_frame).cartesian.xyz.to(
                au.km).value

            X = make_coord_array(x, k, t[i:i+1, None], ref_ant_x[None,:], flat=True)

            X1.x.append(X[:, 0:3])
            X1.k.append(X[:, 3:6])
            X1.t.append(X[:, 6:7])
            X1.ref_x.append(X[:, 7:8])

        X1 = X1._replace(x=jnp.concatenate(X1.x, axis=0),
                         k=jnp.concatenate(X1.k, axis=0),
                         t=jnp.concatenate(X1.t, axis=0),
                         ref_x=jnp.concatenate(X1.ref_x, axis=0),
                         )

        logger.info(f"Total number of coordinates: {X1.x.shape[0]}")

        def compute_covariance_row(X1: GeodesicTuple, X2: GeodesicTuple):
            K = self._kernel(X1, X2, self._bottom, self._width, self._fed_sigma, self._fed_kernel_params,
                             wind_velocity=self._wind_vector)  # 1, N
            return K[0, :]
        covariance_row = lambda X: compute_covariance_row(tree_map(lambda x: x.reshape((1, -1)), X), X1)

        mean = jit(lambda X1: self._kernel.mean_function(X1, self._bottom, self._width, self._fed_mu,
                                                         wind_velocity=self._wind_vector))(X1)

        cov = chunked_pmap(covariance_row, X1, batch_size=X1.x.shape[0], chunksize=ncpu)

        plt.imshow(cov)
        plt.show()


        Z = random.normal(random.PRNGKey(42), (cov.shape[0], 1), dtype=cov.dtype)

        t0 = default_timer()
        jitter = 1e-6
        logger.info(f"Computing Cholesky with jitter: {jitter}")
        L = jnp.linalg.cholesky(cov + jitter * jnp.eye(cov.shape[0]))
        if np.any(np.isnan(L)):
            logger.info("Numerically instable. Using SVD.")
            L = msqrt(cov)

        logger.info(f"Cholesky took {default_timer() - t0} seconds.")

        dtec = (L @ Z + mean[:, None])[:, 0].reshape((Na, Nd, Nt)).transpose((1,0,2))

        logger.info(f"Saving result to {output_h5parm}")
        with dp:
            dp.current_solset = 'sol000'
            dp.select(pol=slice(0, 1, 1))
            dp.tec = np.asarray(dtec[None])

def msqrt(A):

    """
    Computes the matrix square-root using SVD, which is robust to poorly conditi
oned covariance matrices.
    Computes, M such that M @ M.T = A

    Args:
    A: [N,N] Square matrix to take square root of.

    Returns: [N,N] matrix.
    """
    U, s, Vh = jnp.linalg.svd(A)
    L = U * jnp.sqrt(s)
    return L

def main(output_h5parm, ncpu, phase_tracking,
         array_name, start_time, time_resolution, duration,
         field_of_view_diameter, avg_direction_spacing, east_wind, north_wind):

    """
    We sequentially simulate a sequence of Gaussian random variables conditioning off the other variables with a distance
    of screening_distance. We use line-segement distance to compute distance.

    Args:
        output_h5parm:
        ncpu:
        ra:
        dec:
        array_name:
        start_time:
        time_resolution:
        duration:
        field_of_view_diameter:
        avg_direction_spacing:
        east_wind:
        north_wind:
        time_block_size:
        scheduled:

    Returns:

    """
    wind_vector = jnp.asarray([east_wind, north_wind, 0.]) / 1000.  # km/s at 300km height

    sim = Simulation(wind_vector, bottom=200., width=200., l=5., fed_mu=50., fed_sigma=0.6)
    sim.run(output_h5parm,ncpu, avg_direction_spacing,field_of_view_diameter,duration,time_resolution,start_time,array_name,phase_tracking)


def debug_main():
    phase_tracking = ac.SkyCoord("00h00m0.0s","+37d07m47.400s", frame='icrs')
    main(output_h5parm='dsa2000W_2000m_datapack.h5',
         ncpu=64,
         phase_tracking=phase_tracking,
         array_name='dsa2000W_2000m_grid',
         start_time=at.Time('2019-03-19T19:58:14.9', format='isot'),
         time_resolution=60.,
         duration=15*60.,
         field_of_view_diameter=4.,
         avg_direction_spacing=32.,
         east_wind=-242.,
         north_wind=30.)

def add_args(parser):
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.register("type", "phase_tracking", lambda v: ac.SkyCoord(v.split(" "), frame='icrs'))
    parser.register("type", "start_time", lambda v: at.Time(v, format='isot'))
    parser.add_argument('--output_h5parm', help='H5Parm file to file to place the simulated differential TEC, ".h5"',
                        default=None, type=str, required=True)
    parser.add_argument('--phase_tracking', help='Phase tracking center in ICRS frame in format "00h00m0.0s +37d07m47.400s".',
                        default=None, type="phase_tracking", required=True)
    parser.add_argument('--array_name', help=f'Name of array, options are {sorted(list(ARRAYS.keys()))}.',
                        default='dsa2000W_1000m_grid', type=str, required=True)
    parser.add_argument('--start_time', help=f'Start time in isot format "2019-03-19T19:58:14.9".',
                        default=None, type='start_time', required=True)
    parser.add_argument('--time_resolution', help=f'Temporal resolution in seconds.',
                        default=30., type=float, required=False)
    parser.add_argument('--duration', help=f'Temporal resolution in seconds.',
                        default=0., type=float, required=False)
    parser.add_argument('--field_of_view_diameter', help=f'Diameter of field of view in degrees.',
                        default=4., type=float, required=False)
    parser.add_argument('--avg_direction_spacing', help=f'Average spacing between directions in arcmin.',
                        default=32., type=float, required=False)
    parser.add_argument('--east_wind', help=f'Velocity of wind to the east at 100km in m/s.',
                        default=-200., type=float, required=False)
    parser.add_argument('--north_wind', help=f'Velocity of wind to the north at 100km in m/s.',
                        default=0., type=float, required=False)
    parser.add_argument('--ncpu', help='Number of CPUs to use to compute covariance matrix.',
                        default=None, type=int, required=False)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        debug_main()
        exit(0)
    parser = argparse.ArgumentParser(
        description='Simulates DTEC over an observation.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_args(parser)
    flags, unparsed = parser.parse_known_args()
    logger.info("Running with:")
    for option, value in vars(flags).items():
        logger.info("\t{} -> {}".format(option, value))
    main(**vars(flags))