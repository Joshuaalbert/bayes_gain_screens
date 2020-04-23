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
    tp_nn,tn_nn,fp_nn,fn_nn = [],[],[],[]
    tp_gt,tn_gt,fp_gt,fn_gt = [],[],[],[]

    for datapack in datapacks:
        print("Running {}".format(datapack))
        click_data = os.path.join('/home/albert/git/bayes_gain_screens/debug/outlier_detection_adjusted_2/click',os.path.basename(datapack.replace('.h5','.labels.npy')))
        flags = np.load(click_data)#Nd, Na, Nt
        ignore = flags == -1
        ground_truth = np.where(flags == 1, True, False)
        dp = DataPack(datapack, readonly=True)
        dp.select(pol=0)
        dp.current_solset = 'directionally_referenced'
        tec_uncert, axes = dp.weights_tec
        nn_flags = np.isinf(tec_uncert)
        tec, axes = dp.tec
        _, directions = dp.get_directions(axes['dir'])
        Npol, Nd, Na, Nt = tec.shape
        r_flag_file = click_data.replace('.npy','reinout.npy')
        if not os.path.isfile(r_flag_file):
            reinout_flags = np.zeros_like(nn_flags)
            for a in range(Na):
                for t in range(Nt):
                    reinout_flags[0,:,a,t] = reinout_filter(directions.ra.deg, directions.dec.deg, tec[0,:, a, t])
                print("Done {}/{}".format(a+1, Na))
            np.save(r_flag_file, reinout_flags)
        reinout_flags = np.load(r_flag_file)

        tp_nn.append(np.sum(np.logical_and(nn_flags, reinout_flags)))
        tn_nn.append(np.sum(np.logical_and(~nn_flags, ~reinout_flags)))
        fp_nn.append(np.sum(np.logical_and(~nn_flags, reinout_flags)))
        fn_nn.append(np.sum(np.logical_and(nn_flags, ~reinout_flags)))

        tp_gt.append(np.sum(np.logical_and(~ignore, ground_truth, reinout_flags[0, :, :, :])))
        tn_gt.append(np.sum(np.logical_and(~ignore, ~ground_truth, ~reinout_flags[0, :, :, :])))
        fp_gt.append(np.sum(np.logical_and(~ignore, ~ground_truth, reinout_flags[0, :, :, :])))
        fn_gt.append(np.sum(np.logical_and(~ignore, ground_truth, ~reinout_flags[0, :, :, :])))

    tp = np.array(tp_nn).astype(float)
    tn = np.array(tn_nn).astype(float)
    fp = np.array(fp_nn).astype(float)
    fn = np.array(fn_nn).astype(float)

    tpr = tp/(tp + fn)
    fpr = fp/(fp + tn)
    fnr = fn/(fn+tp)
    tnr = tn/(tn + fp)
    acc = (tn + tp)/(tp+tn+fp+fn)

    print('Reinout vs NN (per observation)')
    print(f"TPR: {tpr}")
    print(f"FPR: {fpr}")
    print(f"TNR: {tnr}")
    print(f"FNR: {fnr}")
    print(f"ACC: {acc}")
    tp = tp.sum()
    tn = tn.sum()
    fp = fp.sum()
    fn = fn.sum()

    print('Reinout vs NN (aggregate)')
    print(f"TPR: {tpr}")
    print(f"FPR: {fpr}")
    print(f"TNR: {tnr}")
    print(f"FNR: {fnr}")
    print(f"ACC: {acc}")




    tp = np.array(tp_gt).astype(float)
    tn = np.array(tn_gt).astype(float)
    fp = np.array(fp_gt).astype(float)
    fn = np.array(fn_gt).astype(float)

    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    tnr = tn / (tn + fp)
    acc = (tn + tp) / (tp + tn + fp + fn)

    print('Reinout vs Ground truth (per observation)')
    print(f"TPR: {tpr}")
    print(f"FPR: {fpr}")
    print(f"TNR: {tnr}")
    print(f"FNR: {fnr}")
    print(f"ACC: {acc}")
    tp = tp.sum()
    tn = tn.sum()
    fp = fp.sum()
    fn = fn.sum()

    print('Reinout vs Ground truth (aggregate)')
    print(f"TPR: {tpr}")
    print(f"FPR: {fpr}")
    print(f"TNR: {tnr}")
    print(f"FNR: {fnr}")
    print(f"ACC: {acc}")

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
