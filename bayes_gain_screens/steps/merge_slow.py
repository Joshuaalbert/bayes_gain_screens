import os
import numpy as np
import jax.numpy as jnp
from bayes_gain_screens.utils import great_circle_sep, link_overwrite
from h5parm import DataPack
from h5parm.utils import make_soltab
import argparse
import logging

logger = logging.getLogger(__name__)


def main(data_dir, working_dir, obs_num):
    logger.info("Merging slow solutions into screen and smoothed.")

    smoothed_h5parm = os.path.join(data_dir, 'L{}_DDS5_full_merged.h5'.format(obs_num))

    screen_h5parm = os.path.join(data_dir, 'L{}_DDS6_full_merged.h5'.format(obs_num))

    slow_h5parm = os.path.join(data_dir, 'L{}_DDS7_full_slow_merged.h5'.format(obs_num))

    merged_h5parm = os.path.join(working_dir, 'L{}_DDS8_full_merged.h5'.format(obs_num))
    linked_merged_h5parm = os.path.join(data_dir, os.path.basename(merged_h5parm))
    link_overwrite(merged_h5parm, linked_merged_h5parm)

    select = dict(pol=slice(0, 1, 1))

    ###
    # get slow phase and amplitude

    datapack_slow = DataPack(slow_h5parm, readonly=True)
    logger.info("Getting slow000/phase000+amplitude000")
    datapack_slow.current_solset = 'sol000'
    datapack_slow.select(**select)
    axes = datapack_slow.axes_phase
    patch_names, directions = datapack_slow.get_directions(axes['dir'])
    directions_slow = np.stack([directions.ra.rad, directions.dec.rad], axis=1)
    timestamps, times = datapack_slow.get_times(axes['time'])
    time_slow = times.mjd * 86400.
    phase_slow, axes = datapack_slow.phase
    amplitude_slow, axes = datapack_slow.amplitude

    ###
    # get smoothed phase and amplitude

    datapack_smoothed = DataPack(smoothed_h5parm, readonly=True)
    logger.info("Getting directionally_referenced/const000")
    datapack_smoothed.current_solset = 'sol000'
    datapack_smoothed.select(**select)
    axes = datapack_smoothed.axes_phase
    patch_names, directions = datapack_smoothed.get_directions(axes['dir'])
    directions_smoothed = np.stack([directions.ra.rad, directions.dec.rad], axis=1)
    timestamps, times = datapack_smoothed.get_times(axes['time'])
    time_smoothed = times.mjd * 86400.
    phase_smoothed, axes = datapack_smoothed.phase
    amplitude_smoothed, axes = datapack_smoothed.amplitude
    Ncal = directions_smoothed.shape[0]

    ###
    # get screen phase and amplitude

    datapack_screen = DataPack(screen_h5parm, readonly=False)
    logger.info("Getting screen_posterior/phase000+amplitude000")
    datapack_screen.current_solset = 'sol000'
    datapack_screen.select(**select)
    axes = datapack_screen.axes_phase
    patch_names, directions = datapack_screen.get_directions(axes['dir'])
    directions_screen = np.stack([directions.ra.rad, directions.dec.rad], axis=1)
    timestamps, times = datapack_screen.get_times(axes['time'])
    time_screen = times.mjd * 86400.
    phase_screen, axes = datapack_screen.phase
    amplitude_screen, axes = datapack_screen.amplitude

    ###
    # Create and set screen_slow000

    logger.info("Creating screen_slow/phase000+amplitude000")
    make_soltab(screen_h5parm, from_solset='sol000', to_solset='screen_slow', from_soltab='phase000',
                to_soltab=['phase000', 'amplitude000'], remake_solset=True, to_datapack=merged_h5parm)
    logger.info("Creating smoothed_slow/phase000+amplitude000")
    make_soltab(smoothed_h5parm, from_solset='sol000', to_solset='smoothed_slow', from_soltab='phase000',
                to_soltab=['phase000', 'amplitude000'], remake_solset=True, to_datapack=merged_h5parm)

    logger.info("Creating time mapping")
    time_map = np.asarray([np.argmin(np.abs(time_slow - t)) for t in time_screen])
    logger.info("Creating direction mapping")
    dir_map = np.asarray([jnp.argmin(great_circle_sep(directions_slow[:, 0], directions_slow[:, 1], ra, dec))
                        for (ra, dec) in zip(directions_screen[:, 0], directions_screen[:, 1])])
    #TODO: see if only applying slow on calibrators and screen elsewhere gets rid of artefacts.
    #TODO: see if only applying tec screen gets rid of artefacts (include slow if the above experiment doesn't work)
    #TODO: visibility flagging based on tec outliers (update imaging command)
    phase_smooth_slow = phase_slow[..., time_map] + phase_smoothed
    amplitude_smooth_slow = amplitude_slow[..., time_map] * amplitude_smoothed

    phase_screen_slow = phase_screen + phase_slow[..., time_map][:, dir_map, ...]
    amplitude_screen_slow = amplitude_screen * amplitude_slow[..., time_map][:, dir_map, ...]

    logger.info("Phase screen+slow contains {} nans.".format(np.isnan(phase_screen_slow).sum()))
    phase_screen_slow = np.where(np.isnan(phase_screen_slow), 0., phase_screen_slow)
    logger.info("Amplitude screen+slow contains {} nans.".format(np.isnan(amplitude_screen_slow).sum()))
    amplitude_screen_slow = np.where(np.isnan(amplitude_screen_slow), 1., amplitude_screen_slow)

    logger.info("Saving results to {}".format(merged_h5parm))
    datapack = DataPack(merged_h5parm, readonly=False)
    datapack.current_solset = 'screen_slow'
    datapack.select(**select)
    datapack.phase = phase_screen_slow
    datapack.amplitude = amplitude_screen_slow

    datapack.current_solset = 'smoothed_slow'
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
    logger.info("Running with:")
    for option, value in vars(flags).items():
        logger.info("\t{} -> {}".format(option, value))
    main(**vars(flags))
