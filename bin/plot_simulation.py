import argparse
import sys
import logging
logger = logging.getLogger(__name__)

from bayes_gain_screens.utils import wrap
from bayes_gain_screens.plotting import plot_vornoi_map
from bayes_gain_screens.frames import ENU
from h5parm import DataPack
import astropy.units as au
import astropy.coordinates as ac
import pylab as plt
import numpy as np

class Plot(object):
    def __init__(self, input_datapack):
        self._input_datapack = input_datapack

    def visualise_grid(self, num_directions, num_antennas):
        with DataPack(self._input_datapack, readonly=True) as dp:
            dp.current_solset = 'sol000'
            dp.select(pol=slice(0, 1, 1))
            axes = dp.axes_tec
            patch_names, directions_grid = dp.get_directions(axes['dir'])
            antenna_labels, antennas_grid = dp.get_antennas(axes['ant'])


        dirs = np.random.choice(len(patch_names),size=num_directions, replace=False)
        for d in dirs:
            with DataPack(self._input_datapack, readonly=True) as dp:
                dp.current_solset = 'sol000'
                dp.select(pol=slice(0, 1, 1), dir=slice(d,d+1,1),time=0)
                tec_grid, axes = dp.tec
                tec_grid = tec_grid[0]
                patch_names, directions_grid = dp.get_directions(axes['dir'])
                antenna_labels, antennas_grid = dp.get_antennas(axes['ant'])
                ref_ant = antennas_grid[0]
                timestamps, times_grid = dp.get_times(axes['time'])
                frame = ENU(location=ref_ant.earth_location, obstime=times_grid[0])
                antennas_grid = ac.ITRS(*antennas_grid.cartesian.xyz, obstime=times_grid[0]).transform_to(frame)

            ant_pos = antennas_grid.cartesian.xyz.to(au.km).value.T
            plt.scatter(ant_pos[:,0], ant_pos[:,1],c=tec_grid[0,:,0], cmap=plt.cm.PuOr)
            plt.xlabel('East [km]')
            plt.ylabel("North [km]")
            plt.title(f"Direction {repr(directions_grid)}")
            plt.show()



        ant_scatter_args = (ant_pos[:,0], ant_pos[:,1], tec_grid[0,:,0])
        ants = np.random.choice(len(antenna_labels),size=num_antennas, replace=False)
        for a in ants:
            with DataPack(self._input_datapack, readonly=True) as dp:
                dp.current_solset = 'sol000'
                dp.select(pol=slice(0, 1, 1), ant=slice(a,a+1,1),time=0)
                tec_grid, axes = dp.tec
                tec_grid = tec_grid[0]
                patch_names, directions_grid = dp.get_directions(axes['dir'])
                antenna_labels, antennas_grid = dp.get_antennas(axes['ant'])
                timestamps, times_grid = dp.get_times(axes['time'])
                frame = ENU(location=ref_ant.earth_location, obstime=times_grid[0])
                antennas_grid = ac.ITRS(*antennas_grid.cartesian.xyz, obstime=times_grid[0]).transform_to(frame)

            _ant_pos = antennas_grid.cartesian.xyz.to(au.km).value.T[0]


            fig, axs = plt.subplots(2,1,figsize=(4,8))
            axs[0].scatter(*ant_scatter_args[0:2],c=ant_scatter_args[2], cmap=plt.cm.PuOr,alpha=0.5)
            axs[0].scatter(*_ant_pos[0:2], marker='x',c='red')
            axs[0].set_xlabel('East [km]')
            axs[0].set_ylabel('North [km]')
            pos = 180/np.pi*np.stack([wrap(directions_grid.ra.rad), wrap(directions_grid.dec.rad)],axis=-1)
            plot_vornoi_map(pos,tec_grid[:,0,0], fov_circle=True, ax=axs[1])
            axs[1].set_xlabel('RA(2000) [ded]')
            axs[1].set_ylabel('DEC(2000) [ded]')
            plt.show()

def main(input_h5parm,
         num_directions,
         num_antennas):

    Plot(input_h5parm).visualise_grid(num_directions=num_directions,
                                      num_antennas=num_antennas)


def debug_main():

    main(input_h5parm="/home/albert/git/bayes_gain_screens/bin/dsa2000W_2000m_datapack.h5",
         num_antennas=5,
         num_directions=5)



def add_args(parser):
    parser.add_argument('--h5parm', help='H5Parm file to file to visualise DTEC, ".h5"',
                        default=None, type=str, required=True)
    parser.add_argument('--num_directions', help='Number of directions to plot',
                        default=5, type=int, required=False)
    parser.add_argument('--num_antennas', help='Number of antennas to plot',
                        default=5, type=int, required=False)

if __name__ == '__main__':
    if len(sys.argv) == 1:
        debug_main()
        exit(0)
    parser = argparse.ArgumentParser(
        description='Visualise DTEC over an observation.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_args(parser)
    flags, unparsed = parser.parse_known_args()
    logger.info("Running with:")
    for option, value in vars(flags).items():
        logger.info("\t{} -> {}".format(option, value))
    main(**vars(flags))
