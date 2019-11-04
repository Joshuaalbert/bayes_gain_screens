import os
os.environ['OMP_NUM_THREADS'] = "1"
import numpy as np
import pylab as plt
from scipy.interpolate import griddata
from scipy.optimize import brute, minimize
from scipy.ndimage import median_filter
from bayes_gain_screens import logging
from bayes_gain_screens.datapack import DataPack
from bayes_gain_screens.misc import make_soltab
from bayes_gain_screens.plotting import animate_datapack
from dask.multiprocessing import get
from scipy.spatial import cKDTree
import argparse
from timeit import default_timer

"""
This script is still being debugged/tested. 
Get's TEC from gains.
"""

def main(data_dir, working_dir, obs_num):
    os.chdir(working_dir)
    logging.info("Merging slow solutions into screen and smoothed.")
    original_h5parm = os.path.join(data_dir, 'L{}_{}_merged.h5'.format(obs_num, 'DDS4_full'))
    slow_h5parm = os.path.join(data_dir, 'L{}_{}_merged.h5'.format(obs_num, 'DDS4_full_slow'))
    select = dict(pol = slice(0, 1, 1))

    datapack = DataPack(slow_h5parm, readonly=False)
    logging.info("Getting slow000/phase000+amplitude000")
    datapack.current_solset = 'sol000'
    datapack.select(**select)
    axes = datapack.axes_phase
    patch_names, directions = datapack.get_directions(axes['dir'])
    directions_slow = np.stack([directions.ra.rad*np.cos(directions.dec.rad), directions.dec.rad], axis=1)
    timestamps, times = datapack.get_times(axes['time'])
    time_slow = times.mjd
    phase_slow, axes = datapack.phase
    amplitude_slow, axes = datapack.amplitude


    datapack = DataPack(original_h5parm, readonly=False)

    logging.info("Getting screen_posterior/phase000+amplitude000")
    datapack.current_solset = 'screen_posterior'
    datapack.select(**select)
    axes = datapack.axes_phase
    patch_names, directions = datapack.get_directions(axes['dir'])
    directions_screen = np.stack([directions.ra.rad * np.cos(directions.dec.rad), directions.dec.rad], axis=1)
    timestamps, times = datapack.get_times(axes['time'])
    time_screen = times.mjd
    phase_screen, axes = datapack.phase
    amplitude_screen, axes = datapack.amplitude
    logging.info("Creating screen_slow000/phase000+amplitude000")
    make_soltab(datapack, from_solset='screen_posterior', to_solset='screen_slow000', from_soltab='phase000',
                to_soltab=['phase000', 'amplitude000'], remake_solset=True)
    datapack.current_solset = 'screen_slow000'
    datapack.select(**select)
    _, dir_idx = cKDTree(directions_slow).query(directions_screen)
    _, time_idx = cKDTree(time_slow[:, None]).query(time_screen[:, None])
    phase_screen_slow = phase_screen + phase_slow[:,dir_idx,...][...,time_idx]
    amplitude_screen_slow = amplitude_screen * amplitude_slow[:,dir_idx,...][...,time_idx]
    datapack.phase = phase_screen_slow
    datapack.amplitude = amplitude_screen_slow

    logging.info("Getting smoothed000/phase000+amplitude000")
    datapack.current_solset = 'smoothed000'
    datapack.select(**select)
    axes = datapack.axes_phase
    patch_names, directions = datapack.get_directions(axes['dir'])
    directions_smoothed = np.stack([directions.ra.rad * np.cos(directions.dec.rad), directions.dec.rad], axis=1)
    timestamps, times = datapack.get_times(axes['time'])
    time_smoothed = times.mjd
    phase_smoothed, axes = datapack.phase
    amplitude_smoothed, axes = datapack.amplitude
    logging.info("Creating smoothed_slow000/phase000+amplitude000")
    make_soltab(datapack, from_solset='smoothed000', to_solset='smoothed_slow000', from_soltab='phase000',
                to_soltab=['phase000', 'amplitude000'],remake_solset=True)
    datapack.current_solset = 'smoothed_slow000'
    datapack.select(**select)
    _, dir_idx = cKDTree(directions_slow).query(directions_smoothed)
    _, time_idx = cKDTree(time_slow[:, None]).query(time_smoothed[:, None])
    phase_smoothed_slow = phase_smoothed + phase_slow[:, dir_idx, ...][..., time_idx]
    amplitude_smoothed_slow = amplitude_smoothed * amplitude_slow[:, dir_idx, ...][..., time_idx]
    datapack.phase = phase_smoothed_slow
    datapack.amplitude = amplitude_smoothed_slow


def add_args(parser):
    parser.add_argument('--obs_num', help='Obs number L*',
                        default=None, type=int, required=True)
    parser.add_argument('--data_dir', help='Where are the ms files are stored.',
                        default=None, type=str, required=True)
    parser.add_argument('--working_dir', help='Where to perform the work.',
                        default=None, type=str, required=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Merge slow sols into screen and smoothed in the original smoothed and screen h5parm.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_args(parser)
    flags, unparsed = parser.parse_known_args()
    print("Running with:")
    for option, value in vars(flags).items():
        print("\t{} -> {}".format(option, value))
    main(**vars(flags))
