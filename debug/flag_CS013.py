"""
Flags visibilities based on outlier detection
"""
import matplotlib
matplotlib.use('Agg')
import numpy as np
import argparse
import os
import glob
import pyrap.tables as pt


def main(data_dir, working_dir, obs_num):
    msfiles = glob.glob(os.path.join(data_dir, 'L{}*.ms'.format(obs_num)))
    if len(msfiles) == 0:
        raise IOError("No msfiles")
    mslist_file = os.path.join(working_dir, 'mslist.txt')
    with open(mslist_file, 'w') as f:
        for ms in msfiles:
            f.write("{}\n".format(ms))

    antennas = np.array(['CS013HBA0','CS013HBA1'])

    for ms in msfiles:
        with pt.table(os.path.join(ms, "SPECTRAL_WINDOW")) as t_sw:
            ms_freq = np.mean(t_sw.getcol("CHAN_FREQ"))
        with pt.table(os.path.join(ms, "ANTENNA")) as t_ant:
            ant_names = np.array(t_ant.getcol('NAME'))
            ant_names = ant_names.astype(antennas.dtype)
        flag_ants = np.array([list(ant_names).index(a) for a in antennas])
        with pt.table(ms, readonly=False) as t:
            weights_col = t.getcol("OUTLIER_FLAGS")
            print("Weight col is shape {}".format(weights_col.shape))

            flag_ant1 = np.isin(t.getcol('ANTENNA1'), flag_ants)
            flag_ant2 = np.isin(t.getcol('ANTENNA2'), flag_ants)
            new_flags = np.logical_or(flag_ant1,flag_ant2)
            new_weights = np.where(new_flags[:, None], 0., weights_col)
            t.putcol("OUTLIER_FLAGS", new_weights)
            print("Stored flags in {}".format("OUTLIER_FLAGS"))


def add_args(parser):
    parser.add_argument('--obs_num', help='Obs number L*',
                        default=None, type=int, required=True)
    parser.add_argument('--data_dir', help='Where are the ms files are stored.',
                        default=None, type=str, required=True)
    parser.add_argument('--working_dir', help='Where to perform the imaging.',
                        default=None, type=str, required=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Flag CS013.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_args(parser)
    flags, unparsed = parser.parse_known_args()
    print("Running with:")
    for option, value in vars(flags).items():
        print("\t{} -> {}".format(option, value))
    main(**vars(flags))
