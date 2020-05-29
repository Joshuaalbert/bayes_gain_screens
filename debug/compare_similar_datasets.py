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
    for dp in dps:
        dp.current_solset = solset
    correct = np.ones((len(dps),len(dps)), dtype=np.bool)
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
        print("{} -> {} {}:{}".format(i, f, solset, soltab))
        print("\t".join(h5parms))
    print("Similarity matrix:")
    print(correct)
    return correct


def assert_same(correct,msg=""):
    if not np.all(correct):
        print(f"ASSERT failed: {msg} not the same")
        return False
    return True



def assert_different(correct,msg=""):
    if not (np.sum(correct) == np.sum(np.diag(correct))):
        print(f"ASSERT failed: {msg} not different")
        return False
    return True


def main(root_paths):
    root_paths = [os.path.abspath(p) for p in root_paths]
    data_dirs = [os.path.join(p, 'download_archive') for p in root_paths]

    print("Comparing initial solutions from the first killMS solve.")
    dds4_h5parms = [glob.glob(os.path.join(p, 'L*_DDS4_full_merged.h5'))[0] for p in data_dirs]
    try:
        dds4_phase_same = assert_same(compare_datapacks(dds4_h5parms, 'sol000', 'phase000', select=dict(pol=0)))
    except FileNotFoundError:
        print("Did not find dds4")
    try:
        dds4_amp_same = assert_same(compare_datapacks(dds4_h5parms, 'sol000', 'amplitude000', select=dict(pol=0)))
    except FileNotFoundError:
        print("Did not find dds4")

    dds4_same = dds4_amp_same and dds4_phase_same
    if not dds4_same:
        print("ERROR: original solutions are not the same between runs, which implies different calibrators were chosen.")

    print("Comparing smoothed000 from tec_inference_and_smooth.")
    dds5_h5parms = [glob.glob(os.path.join(p, 'L*_DDS5_full_merged.h5'))[0] for p in data_dirs]
    try:
        dds5_phase_same = assert_same(compare_datapacks(dds5_h5parms, 'smoothed000', 'phase000', select=dict(pol=0)))
    except FileNotFoundError:
        print("Did not find dds5")
    try:
        dds5_amp_same = assert_same(compare_datapacks(dds5_h5parms, 'smoothed000', 'amplitude000', select=dict(pol=0)))
    except FileNotFoundError:
        print("Did not find dds5")

    dds5_smooth_same = dds5_phase_same and dds5_amp_same
    if dds4_same and not dds5_smooth_same:
        print("ERROR: smoothing should be the same if original solutions were the same but are not. Implies smoothing depends on the difference in parmetrisation between datapacks. Should not be the case unless you are optimising the smoothing step.")

    print("Comparing tec000 and const000 from tec_inference_and_smooth.")
    try:
        dds5_tec_diff = assert_different(compare_datapacks(dds5_h5parms, 'directionally_referenced', 'tec000', select=dict(pol=0)))
    except FileNotFoundError:
        print("Did not find dds5")
    # different const
    try:
        dds5_const_diff = assert_different(compare_datapacks(dds5_h5parms, 'directionally_referenced', 'const000', select=dict(pol=0)))
    except FileNotFoundError:
        print("Did not find dds5")

    dds5_inference_diff = dds5_tec_diff and dds5_const_diff
    if not dds5_inference_diff:
        print("ERROR: tec inference is the same which implies things were run with the same setting on accident")

    print("Comparing screen_posterior phase000, tec000 from infer_screen.")
    dds6_h5parms = [glob.glob(os.path.join(p, 'L*_DDS6_full_merged.h5'))[0] for p in data_dirs]
    try:
        dds6_tec_diff = assert_different(compare_datapacks(dds6_h5parms, 'screen_posterior', 'tec000', select=dict(pol=0)))
    except FileNotFoundError:
        print("Did not find dds6")
    try:
        dds6_phase_diff = assert_different(compare_datapacks(dds6_h5parms, 'screen_posterior', 'phase000', select=dict(pol=0)))
    except FileNotFoundError:
        print("Did not find dds6")

    dds6_diff = dds6_tec_diff and dds6_phase_diff
    if dds5_inference_diff and not dds6_diff:
        print("ERROR: calibrator inference was different but screen posterior is the same. Implies a file mix up.")

    print("Comparing slow solutions, which should be the same if smooth000 was the same.")
    dds7_h5parms = [glob.glob(os.path.join(p, 'L*_DDS7_full_slow_merged.h5'))[0] for p in data_dirs]
    try:
        dds7_phase_same = assert_same(compare_datapacks(dds7_h5parms, 'sol000', 'phase000', select=dict(pol=0)))
    except FileNotFoundError:
        print("Did not find dds7")
    try:
        dds7_amp_same = assert_same(compare_datapacks(dds7_h5parms, 'sol000', 'amplitude000', select=dict(pol=0)))
    except FileNotFoundError:
        print("Did not find dds7")

    dds7_same = dds7_amp_same and dds7_phase_same
    if dds5_smooth_same and not dds7_same:
        print("ERROR: smooth solutions were the same but the slow were different. Means something wrong in preparing the pre-apply solutions!")


    print("Comparing final screen, on calibrators. Should be the same if smoothed000 and slow were the same")
    dds8_h5parms = [glob.glob(os.path.join(p, 'L*_DDS8_full_merged.h5'))[0] for p in data_dirs]
    try:
        dds8_cal_phase_same = assert_same(compare_datapacks(dds8_h5parms, 'screen_slow000', 'phase000', select=dict(pol=0, dir=slice(0, 45, 1))))
    except FileNotFoundError:
        print("Did not find dds8")
    try:
        dds8_cal_amp_same = assert_same(compare_datapacks(dds8_h5parms, 'screen_slow000', 'amplitude000', select=dict(pol=0, dir=slice(0, 45, 1))))
    except FileNotFoundError:
        print("Did not find dds8")

    dds8_cal_same = dds8_cal_amp_same and dds8_cal_phase_same
    if (dds5_smooth_same and dds7_same) and not dds8_cal_same:
        print("ERROR: Final screen cals different, but smooth and slow are the same. Problem in merge step!")

    print("Comparing final screen on non-calibrators. Should be different.")
    try:
        dds8_ncal_phase_diff = assert_different(compare_datapacks(dds8_h5parms, 'screen_slow000', 'phase000', select=dict(pol=0, dir=slice(45, None, 1))))
    except FileNotFoundError:
        print("Did not find dds8")
    try:
        dds8_ncal_amp_diff = assert_different(compare_datapacks(dds8_h5parms, 'screen_slow000', 'amplitude000', select=dict(pol=0, dir=slice(45, None, 1))))
    except FileNotFoundError:
        print("Did not find dds8")
    dds8_ncal_diff = dds8_ncal_amp_diff and dds8_ncal_phase_diff

    if not dds8_ncal_diff:
        print("ERROR: Final screen non-calibrators are the same, which implies a file mixup, or accidental invokation with identical parameters!")


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
