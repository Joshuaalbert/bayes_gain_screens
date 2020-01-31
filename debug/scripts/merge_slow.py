import os
import numpy as np
from bayes_gain_screens import logging
from bayes_gain_screens.datapack import DataPack
from bayes_gain_screens.misc import great_circle_sep
from bayes_gain_screens.misc import make_soltab
import argparse

"""
This script is still being debugged/tested. 
Get's TEC from gains.
"""

def main(data_dir, working_dir, obs_num):
    os.chdir(working_dir)
    logging.info("Merging slow solutions into screen and smoothed.")
    original_h5parm = os.path.join(data_dir, 'L{}_DDS4_full_merged.h5'.format(obs_num))
    slow_h5parm = os.path.join(data_dir, 'L{}_DDS4_full_slow_merged.h5'.format(obs_num))
    select = dict(pol = slice(0, 1, 1))

    ###
    # get slow phase and amplitude

    datapack_slow = DataPack(slow_h5parm, readonly=True)
    logging.info("Getting slow000/phase000+amplitude000")
    datapack_slow.current_solset = 'sol000'
    datapack_slow.select(**select)
    axes = datapack_slow.axes_phase
    patch_names, directions = datapack_slow.get_directions(axes['dir'])
    directions_slow = np.stack([directions.ra.rad, directions.dec.rad], axis=1)
    timestamps, times = datapack_slow.get_times(axes['time'])
    time_slow = times.mjd*86400.
    phase_slow, axes = datapack_slow.phase
    amplitude_slow, axes = datapack_slow.amplitude


    ###
    # get screen phase and amplitude

    datapack = DataPack(original_h5parm, readonly=False)
    logging.info("Getting screen_posterior/phase000+amplitude000")
    datapack.current_solset = 'screen_posterior'
    datapack.select(**select)
    axes = datapack.axes_tec
    patch_names, directions = datapack.get_directions(axes['dir'])
    directions_screen = np.stack([directions.ra.rad, directions.dec.rad], axis=1)
    timestamps, times = datapack.get_times(axes['time'])
    time_screen = times.mjd*86400.
    phase_screen, axes = datapack.phase
    amplitude_screen, axes = datapack.amplitude

    ###
    # get smoothed000 phase and amplitude

    logging.info("Getting smoothed000/phase000+amplitude000")
    datapack.current_solset = 'smoothed000'
    datapack.select(**select)
    axes = datapack.axes_phase
    patch_names, directions = datapack.get_directions(axes['dir'])
    directions_smoothed = np.stack([directions.ra.rad, directions.dec.rad], axis=1)
    timestamps, times = datapack.get_times(axes['time'])
    time_smoothed = times.mjd * 86400.
    phase_smoothed, axes = datapack.phase
    amplitude_smoothed, axes = datapack.amplitude
    Ncal = directions_smoothed.shape[0]

    ###
    # Create and set screen_slow000

    logging.info("Creating screen_slow000/phase000+amplitude000")
    make_soltab(datapack, from_solset='screen_posterior', to_solset='screen_slow000', from_soltab='phase000',
                to_soltab=['phase000', 'amplitude000'], remake_solset=True)
    logging.info("Creating smoothed_slow000/phase000+amplitude000")
    make_soltab(datapack, from_solset='smoothed000', to_solset='smoothed_slow000', from_soltab='phase000',
                to_soltab=['phase000', 'amplitude000'], remake_solset=True)

    logging.info("Creating time mapping")
    time_map = np.array([np.argmin(np.abs(time_slow - t)) for t in time_screen])
    logging.info("Creating direction mapping")
    dir_map = np.array([np.argmin(great_circle_sep(directions_slow[:,0], directions_slow[:,1], ra, dec))
                        for (ra, dec) in zip(directions_screen[:,0], directions_screen[:, 1])])

    phase_smooth_slow = phase_slow[..., time_map] + phase_smoothed
    amplitude_smooth_slow = amplitude_slow[..., time_map] * amplitude_smoothed

    phase_screen_slow = phase_screen # + phase_slow[..., time_map][:, dir_map, ...] #Don't add slow to screen
    phase_screen_slow[:, :Ncal, ...] = phase_smooth_slow
    # Amplitudes are fit with rbf during deploy, so we can keep those or replace with NN here
    amplitude_screen_slow = amplitude_smooth_slow[:, dir_map, ...] #amplitude_smoothed[:, dir_map, ...] #amplitude_screen # * amplitude_smooth_slow[:, dir_map, ...]

    datapack.current_solset = 'screen_slow000'
    datapack.select(**select)
    datapack.phase = phase_screen_slow
    datapack.amplitude = amplitude_screen_slow

    datapack.current_solset = 'smoothed_slow000'
    datapack.select(**select)
    datapack.phase = phase_smooth_slow
    datapack.amplitude = amplitude_smooth_slow


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
