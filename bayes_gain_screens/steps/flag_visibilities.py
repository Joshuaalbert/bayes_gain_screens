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
import pylab as plt
import tables


def main(data_dir, working_dir, obs_num, new_weights_col, outlier_frac_thresh):
    msfiles = glob.glob(os.path.join(data_dir, 'L{}*.ms'.format(obs_num)))
    if len(msfiles) == 0:
        raise IOError("No msfiles")
    mslist_file = os.path.join(working_dir, 'mslist.txt')
    with open(mslist_file, 'w') as f:
        for ms in msfiles:
            f.write("{}\n".format(ms))
    merged_h5parm = os.path.join(data_dir, 'L{}_{}_merged.h5'.format(obs_num, 'DDS4_full'))
    with tables.open_file(merged_h5parm) as datapack:
        root = getattr(datapack, "root")
        directionally_referenced = getattr(root, "directionally_referenced")
        tec_soltab = getattr(directionally_referenced, "tec000")
        times = tec_soltab.time[:]
        antennas = np.array(tec_soltab.ant[:])
        # Npol, Nd, Na, Nt
        tec_uncert = tec_soltab.weight[...]
        Npol, Nd, Na, Nt = tec_uncert.shape
        outliers = np.isinf(tec_uncert)
    # Na, Nt
    flags = np.sum(outliers[0, ...], axis=0) > Nd * outlier_frac_thresh

    #plotting some things
    frac_flagged = np.sum(flags, axis=0)/float(Na)
    plt.plot(frac_flagged)
    plt.xlabel('Time')
    plt.ylabel("Frac flagged [1]")
    plt.ylim(0., 1.)
    plt.savefig(os.path.join(working_dir, "frac_flagged_per_time.png"))
    plt.savefig(os.path.join(working_dir, "frac_flagged_per_time.pdf"))
    plt.close('all')

    frac_flagged = np.sum(flags, axis=1) / float(Nt)
    plt.plot(frac_flagged)
    plt.xlabel('Antenna index')
    plt.ylabel("Frac flagged [1]")
    plt.ylim(0., 1.)
    plt.savefig(os.path.join(working_dir, "frac_flagged_per_ant.png"))
    plt.savefig(os.path.join(working_dir, "frac_flagged_per_ant.pdf"))
    plt.close('all')

    frac_outliers = np.sum(outliers[0, ...], axis=0)/float(Nd)
    frac_flagged = np.sum(frac_outliers, axis=0) / float(Na)
    plt.plot(frac_flagged)
    plt.xlabel('Time')
    plt.ylabel("Frac flagged [1]")
    plt.ylim(0., 1.)
    plt.savefig(os.path.join(working_dir, "frac_outliers_per_time.png"))
    plt.savefig(os.path.join(working_dir, "frac_outliers_per_time.pdf"))
    plt.close('all')

    frac_flagged = np.sum(frac_outliers, axis=1) / float(Nt)
    plt.plot(frac_flagged)
    plt.xlabel('Antenna index')
    plt.ylabel("Frac flagged [1]")
    plt.ylim(0., 1.)
    plt.savefig(os.path.join(working_dir, "frac_outliers_per_ant.png"))
    plt.savefig(os.path.join(working_dir, "frac_outliers_per_ant.pdf"))
    plt.close('all')



    for ms in msfiles:
        with pt.table(os.path.join(ms, "SPECTRAL_WINDOW")) as t_sw:
            ms_freq = np.mean(t_sw.getcol("CHAN_FREQ"))
        with pt.table(os.path.join(ms, "ANTENNA")) as t_ant:
            ant_names = np.array(t_ant.getcol('NAME'))
            ant_names = ant_names.astype(antennas.dtype)
        ant_map = np.array([list(antennas).index(a) for a in ant_names])
        with pt.table(ms, readonly=False) as t:
            weights_col = t.getcol("IMAGING_WEIGHT")
            print("Weight col is shape {}".format(weights_col.shape))
            cols = t.colnames()
            if new_weights_col in cols:
                t.removecols(new_weights_col)
            desc = t.getcoldesc("IMAGING_WEIGHT")
            desc['name'] = new_weights_col
            t.addcols(desc)
            print("Created {}".format(new_weights_col))
            # t.putcol(new_weights_col, weights_col)
            # print("Stored original weights")
            vis_ant1 = ant_map[t.getcol('ANTENNA1')]
            vis_ant2 = ant_map[t.getcol('ANTENNA2')]
            vis_times = t.getcol('TIME')
            # indexes closest point in solset
            time_map = np.searchsorted(0.5 * (times[1:] + times[:-1]), vis_times, side='right')
            new_flags = np.logical_or(flags[vis_ant1, time_map], flags[vis_ant2, time_map])
            print("Flagged [{} / {}] baselines ({:.2f}%)".format(np.sum(new_flags), new_flags.size,
                                                                 100. * (np.sum(new_flags) / float(new_flags.size))))
            new_weights = np.where(new_flags[:, None], 0., weights_col)
            t.putcol(new_weights_col, new_weights)
            print("Stored flags in {}".format(new_weights_col))


def add_args(parser):
    parser.add_argument('--obs_num', help='Obs number L*',
                        default=None, type=int, required=True)
    parser.add_argument('--data_dir', help='Where are the ms files are stored.',
                        default=None, type=str, required=True)
    parser.add_argument('--working_dir', help='Where to perform the imaging.',
                        default=None, type=str, required=True)
    parser.add_argument('--new_weights_col', help='Name of weight column to insert flags.',
                        default="OUTLIER_FLAGS", type=str, required=False)
    parser.add_argument('--outlier_frac_thresh', help='What fraction of directions outliers before flagging.',
                        default=30./45., type=float, required=False)


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
