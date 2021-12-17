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

    def visualise_grid(self):
        for d in range(0,5):
            with DataPack(self._input_datapack, readonly=True) as dp:
                dp.current_solset = 'sol000'
                dp.select(pol=slice(0, 1, 1), dir=d,time=0)
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

        for a in [0,50,150,200,250]:
            with DataPack(self._input_datapack, readonly=True) as dp:
                dp.current_solset = 'sol000'
                dp.select(pol=slice(0, 1, 1), ant=a,time=0)
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

def main(input_h5parm):

    Plot(input_h5parm).visualise_grid()


def debug_main():

    main(input_h5parm="/home/albert/data/ionosphere/dsa2000W_datapack.h5")


if __name__ == '__main__':
    if len(sys.argv) == 1:
        debug_main()
        exit(0)