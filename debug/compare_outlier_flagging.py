from bayes_gain_screens.outlier_detection import remove_outliers, Classifier, reinout_filter
from bayes_gain_screens.datapack import DataPack
import numpy as np
import argparse, os, glob

def compare_outlier_methods(datapacks, ref_images, working_dir):
    print("Comparing flagging methods")
    # remove_outliers(False, False, True,
    #                 datapacks,
    #                 ref_images,
    #                 working_dir,
    #                 Classifier.flagging_models,
    #                 K=15,
    #                 L=10,
    #                 n_features=48,
    #                 batch_size=16)
    tp,tn,fp,fn = [],[],[],[]
    for datapack in datapacks:
        print("Running {}".format(datapack))
        dp = DataPack(datapack, readonly=True)
        dp.select(pol=0)
        dp.current_solset = 'directionally_referenced'
        tec_uncert, axes = dp.weights_tec
        nn_flags = np.isinf(tec_uncert)
        tec, axes = dp.tec
        directions = dp.get_directions(axes['dir'])
        Npol, Nd, Na, Nt = tec.shape
        reinout_flags = np.zeros_like(nn_flags)
        for a in range(Na):
            for t in range(Nt):
                reinout_flags[0,:,a,t] = reinout_filter(directions.ra.deg, directions.dec.deg, tec[0,:, a, t])
            print("Done {}/{}".format(a+1, Na))
        tp.append(np.sum(np.logical_and(nn_flags, reinout_flags)))
        tn.append(np.sum(np.logical_and(~nn_flags, ~reinout_flags)))
        fp.append(np.sum(np.logical_and(~nn_flags, reinout_flags)))
        fn.append(np.sum(np.logical_and(nn_flags, ~reinout_flags)))
        np.savez('./outlier_comparison.npz',tp=np.array(tp), tn=np.array(tn), fp=np.array(fp), fn=np.array(fn))

def add_args(parser):
    def string_or_none(s):
        if s.lower() == 'none':
            return None
        else:
            return s
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.register('type', 'str_or_none', string_or_none)

    parser.add_argument('--working_dir', help='Where to store click and model data if doing clicking and training',
                        default=None, type="str_or_none", required=False)
    parser.add_argument('--datapack_pattern', help='Pattern of datapacks.',
                        default=None, type=str, required=False)
    parser.add_argument('--ref_image_pattern', help='Pattern of ref_images.',
                        default=None, type=str, required=False)

def main(working_dir, datapack_pattern, ref_image_pattern):


    if datapack_pattern is None:
        raise ValueError("datapack pattern can't be none if clicking")
    if ref_image_pattern is None:
        raise ValueError("ref image pattern can't be none in clicking")
    datapacks = glob.glob(datapack_pattern)
    ref_images = glob.glob(ref_image_pattern)
    if len(ref_images) == 1 and len(ref_images) != len(datapacks):
        ref_images = ref_images*len(datapacks)

    compare_outlier_methods(datapacks, ref_images, working_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Variational inference of DDTEC and a constant term. Updates the smoothed000 solset too.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_args(parser)
    flags, unparsed = parser.parse_known_args()
    print("Running with:")
    for option, value in vars(flags).items():
        print("    {} -> {}".format(option, value))
    main(**vars(flags))
