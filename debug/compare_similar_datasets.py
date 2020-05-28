from bayes_gain_screens.datapack import DataPack
import os, glob, sys
import numpy as np
import argparse


def compare_datapacks(h5parms, solset, soltab, weight=False, select=None):
    for f in h5parms:
        if not os.path.isfile(f):
            raise FileNotFoundError()

    dps = [DataPack(f, readonly=True) for f in h5parms]
    if select is not None:
        for dp in dps:
            dp.select(**select)
    correct = np.ones(len(dps), dtype=np.bool)
    for i in range(len(dps)):
        for j in range(i + 1, len(dps)):
            dp1 = dps[i]
            dp2 = dps[j]
            data1, axes1 = dp1.get_soltab(soltab, weight=weight)
            data2, axes2 = dp2.get_soltab(soltab, weight=weight)
            # for k, v in axes1.items():
            #     if ~np.all(np.isclose(v, axes2[k])):
            #         print("Axes are not the same of {} and {} on solset {} and soltab {}".format(h5parms[i], h5parms[j],
            #                                                                                      solset, soltab))
            #         correct[i, j] = False
            #         correct[j, i] = False

            if ~np.all(np.isclose(data1, data2)):
                print("Data are not the same of {} and {} on solset {} and soltab {}".format(h5parms[i], h5parms[j],
                                                                                             solset, soltab))
                correct[i, j] = False
                correct[j, i] = False
    for i, f in enumerate(h5parms):
        print("{} -> {}".format(i, f))
        print("\t".join(h5parms))
    print("Same matrix:")
    print(correct)
    return correct


def assert_same(correct):
    assert np.all(correct)


def assert_different(correct):
    assert np.sum(correct) == np.sum(np.diag(correct))


def main(root_paths):
    root_paths = [os.path.abspath(p) for p in root_paths]
    data_dirs = [os.path.join(p, 'download_archive') for p in root_paths]
    # solutions from first Kms solve
    dds4_h5parms = [glob.glob(os.path.join(p, 'L*_DDS4_full_merged.h5'))[0] for p in data_dirs]
    # smoothed000 and directionally_referenced solsets
    dds5_h5parms = [glob.glob(os.path.join(p, 'L*_DDS5_full_merged.h5'))[0] for p in data_dirs]
    # screen_posterior solset, amplitude000, phase000, tec000
    dds6_h5parms = [glob.glob(os.path.join(p, 'L*_DDS6_full_merged.h5'))[0] for p in data_dirs]
    # slow sols, sol000 phase and amplitude
    dds7_h5parms = [glob.glob(os.path.join(p, 'L*_DDS7_full_slow_merged.h5'))[0] for p in data_dirs]
    dds8_h5parms = [glob.glob(os.path.join(p, 'L*_DDS8_full_merged.h5'))[0] for p in data_dirs]

    # same initial solutions
    try:
        assert_same(compare_datapacks(dds4_h5parms, 'sol000', 'phase000', select=dict(pol=0)))
    except FileNotFoundError:
        print("Did not find dds4")
    try:
        assert_same(compare_datapacks(dds4_h5parms, 'sol000', 'amplitude000', select=dict(pol=0)))
    except FileNotFoundError:
        print("Did not find dds4")
    # same smoothed
    try:
        assert_same(compare_datapacks(dds5_h5parms, 'smoothed000', 'phase000', select=dict(pol=0)))
    except FileNotFoundError:
        print("Did not find dds5")
    try:
        assert_same(compare_datapacks(dds5_h5parms, 'smoothed000', 'amplitude000', select=dict(pol=0)))
    except FileNotFoundError:
        print("Did not find dds5")
    # different tec
    try:
        assert_different(compare_datapacks(dds5_h5parms, 'directionally_referenced', 'tec000', select=dict(pol=0)))
    except FileNotFoundError:
        print("Did not find dds5")
    # different const
    try:
        assert_different(compare_datapacks(dds5_h5parms, 'directionally_referenced', 'const000', select=dict(pol=0)))
    except FileNotFoundError:
        print("Did not find dds5")
    # different screens tec-only
    try:
        assert_different(compare_datapacks(dds6_h5parms, 'screen_posterior', 'tec000', select=dict(pol=0)))
    except FileNotFoundError:
        print("Did not find dds6")
    try:
        assert_different(compare_datapacks(dds6_h5parms, 'screen_posterior', 'phase000', select=dict(pol=0)))
    except FileNotFoundError:
        print("Did not find dds6")
    # same slow solutions
    try:
        assert_same(compare_datapacks(dds7_h5parms, 'sol000', 'phase000', select=dict(pol=0)))
    except FileNotFoundError:
        print("Did not find dds7")
    try:
        assert_same(compare_datapacks(dds7_h5parms, 'sol000', 'amplitude000', select=dict(pol=0)))
    except FileNotFoundError:
        print("Did not find dds7")
    # same calibrators in final screen
    try:
        assert_same(compare_datapacks(dds8_h5parms, 'screen_slow000', 'phase000', select=dict(pol=0, dir=slice(0, 45, 1))))
    except FileNotFoundError:
        print("Did not find dds8")
    try:
        assert_same(compare_datapacks(dds8_h5parms, 'screen_slow000', 'amplitude000', select=dict(pol=0, dir=slice(0, 45, 1))))
    except FileNotFoundError:
        print("Did not find dds8")
    # different non-calibrators
    try:
        assert_same(compare_datapacks(dds8_h5parms, 'screen_slow000', 'phase000', select=dict(pol=0, dir=slice(45, None, 1))))
    except FileNotFoundError:
        print("Did not find dds8")
    try:
        assert_same(compare_datapacks(dds8_h5parms, 'screen_slow000', 'amplitude000', select=dict(pol=0, dir=slice(45, None, 1))))
    except FileNotFoundError:
        print("Did not find dds8")


def add_args(parser):
    parser.add_argument('--root_paths', help='List of L<obsnum> folders to compare.', type=str, required=True,
                        nargs='+')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compare two datasets for similarity.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_args(parser)
    flags, unparsed = parser.parse_known_args()
    print("Running with:")
    for option, value in vars(flags).items():
        print("\t{} -> {}".format(option, value))
    main(**vars(flags))
