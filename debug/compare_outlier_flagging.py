from bayes_gain_screens.outlier_detection import remove_outliers, Classifier, reinout_filter
from bayes_gain_screens.datapack import DataPack
import numpy as np
import pylab as plt
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
    tp_r, tn_r, fp_r, fn_r = [],[],[],[]
    tp_thresh, tn_thresh, fp_thresh, fn_thresh = [],[],[],[]
    sizes = []
    num_outliers = []
    dir_thresh = np.arange(45)
    for datapack in datapacks:
        print("Running {}".format(datapack))
        click_data = os.path.join('/home/albert/git/bayes_gain_screens/debug/outlier_detection_adjusted_2/click',os.path.basename(datapack.replace('.h5','.labels.npy')))
        flags = np.load(click_data)#Nd, Na, Nt
        ignore = flags == -1
        print(np.sum(~ignore),'labelled')
        ground_truth = flags == 1
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
        sizes.append(Nd*Na*Nt)
        num_outliers.append(nn_flags.sum())

        tp_nn.append(np.sum(np.logical_and(nn_flags, reinout_flags)))
        tn_nn.append(np.sum(np.logical_and(~nn_flags, ~reinout_flags)))
        fp_nn.append(np.sum(np.logical_and(~nn_flags, reinout_flags)))
        fn_nn.append(np.sum(np.logical_and(nn_flags, ~reinout_flags)))

        tp_gt.append(np.sum(np.logical_and(~ignore,np.logical_and(ground_truth, reinout_flags[0, :, :, :]))))
        tn_gt.append(np.sum(np.logical_and(~ignore,np.logical_and(~ground_truth, ~reinout_flags[0, :, :, :]))))
        fp_gt.append(np.sum(np.logical_and(~ignore,np.logical_and(~ground_truth, reinout_flags[0, :, :, :]))))
        fn_gt.append(np.sum(np.logical_and(~ignore,np.logical_and(ground_truth, ~reinout_flags[0, :, :, :]))))

        tp_r.append(np.sum(np.logical_and(~ignore, np.logical_and(ground_truth, nn_flags[0, :, :, :]))))
        tn_r.append(np.sum(np.logical_and(~ignore, np.logical_and(~ground_truth, ~nn_flags[0, :, :, :]))))
        fp_r.append(np.sum(np.logical_and(~ignore, np.logical_and(~ground_truth, nn_flags[0, :, :, :]))))
        fn_r.append(np.sum(np.logical_and(~ignore, np.logical_and(ground_truth, ~nn_flags[0, :, :, :]))))

        #S, Nd, Na, Nt
        flag_thresh = np.tile((np.sum(nn_flags[0,...], axis=0) > dir_thresh[:, None, None])[:, None, :, :], [1,nn_flags.shape[1], 1, 1])
        flag_thresh = np.logical_or(flag_thresh, nn_flags)
        # S, Nd, Na, Nt
        tp_thresh.append(np.sum(np.logical_and(~ignore, np.logical_and(ground_truth, flag_thresh)).reshape((45, -1)),axis=-1))
        tn_thresh.append(np.sum(np.logical_and(~ignore, np.logical_and(~ground_truth, ~flag_thresh)).reshape((45, -1)),axis=-1))
        fp_thresh.append(np.sum(np.logical_and(~ignore, np.logical_and(~ground_truth, flag_thresh)).reshape((45, -1)),axis=-1))
        fn_thresh.append(np.sum(np.logical_and(~ignore, np.logical_and(ground_truth, ~flag_thresh)).reshape((45, -1)),axis=-1))

    sizes = np.array(sizes)
    num_outliers = np.array(num_outliers)

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

    print("Total outliers found: {}".format(num_outliers))
    print("Total outliers found: {}".format(np.sum(num_outliers)))

    print("Estimated false negatives: {}".format(num_outliers * fnr))
    print("Estimated false negatives: {}".format(np.sum(num_outliers * fnr)))
    print("Estimated false positives: {}".format((sizes - num_outliers) * fpr))
    print("Estimated false positives: {}".format(np.sum((sizes - num_outliers) * fpr)))

    tp = tp.sum()
    tn = tn.sum()
    fp = fp.sum()
    fn = fn.sum()

    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    tnr = tn / (tn + fp)
    acc = (tn + tp) / (tp + tn + fp + fn)

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

    print("Outliers {}".format(np.sum(tp+fn)))
    print("Non-outliers {}".format(np.sum(tn + fp)))

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

    print("Total outliers found: {}".format(num_outliers))
    print("Total outliers found: {}".format(np.sum(num_outliers)))

    print("Estimated false negatives: {}".format(num_outliers * fnr))
    print("Estimated false negatives: {}".format(np.sum(num_outliers * fnr)))
    print("Estimated false positives: {}".format((sizes - num_outliers) * fpr))
    print("Estimated false positives: {}".format(np.sum((sizes - num_outliers) * fpr)))

    FP_reinout = (sizes - num_outliers) * fpr
    FN_reinout = num_outliers * fnr
    FNR_reinout = fnr
    FPR_reinout = fpr



    tp = tp.sum()
    tn = tn.sum()
    fp = fp.sum()
    fn = fn.sum()

    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    tnr = tn / (tn + fp)
    acc = (tn + tp) / (tp + tn + fp + fn)

    print('Reinout vs Ground truth (aggregate)')
    print(f"TPR: {tpr}")
    print(f"FPR: {fpr}")
    print(f"TNR: {tnr}")
    print(f"FNR: {fnr}")
    print(f"ACC: {acc}")

    tFNR_reinout = fnr
    tFPR_reinout = fpr

    tp = np.array(tp_r).astype(float)
    tn = np.array(tn_r).astype(float)
    fp = np.array(fp_r).astype(float)
    fn = np.array(fn_r).astype(float)

    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    tnr = tn / (tn + fp)
    acc = (tn + tp) / (tp + tn + fp + fn)



    print('NN vs Ground truth (per observation)')
    print(f"TPR: {tpr}")
    print(f"FPR: {fpr}")
    print(f"TNR: {tnr}")
    print(f"FNR: {fnr}")
    print(f"ACC: {acc}")

    #P = tp + fn = tp + fnr * P -> tp = P*(1-fnr)
    #N = tn + fp = tn + fpr * N -> tn = N*(1-fpr)

    print("Total outliers found: {}".format(num_outliers))
    print("Total outliers found: {}".format(np.sum(num_outliers)))

    print("Estimated false negatives: {}".format(num_outliers * fnr))
    print("Estimated false negatives: {}".format(np.sum(num_outliers * fnr)))
    print("Estimated false positives: {}".format((sizes - num_outliers) * fpr ))
    print("Estimated false positives: {}".format(np.sum((sizes - num_outliers) * fpr)))

    FP_nn = (sizes - num_outliers) * fpr
    FN_nn = num_outliers * fnr
    FNR_nn = fnr
    FPR_nn = fpr

    tp = tp.sum()
    tn = tn.sum()
    fp = fp.sum()
    fn = fn.sum()

    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    tnr = tn / (tn + fp)
    acc = (tn + tp) / (tp + tn + fp + fn)

    print('NN vs Ground truth (aggregate)')
    print(f"TPR: {tpr}")
    print(f"FPR: {fpr}")
    print(f"TNR: {tnr}")
    print(f"FNR: {fnr}")
    print(f"ACC: {acc}")
    tFNR_nn = fnr
    tFPR_nn = fpr



    for z in list(zip(datapacks, FNR_nn*100, FPR_nn*100, FN_nn, FP_nn, FNR_reinout*100, FPR_reinout*100, FN_reinout, FP_reinout))[::-1]:
        datapack = z[0]
        obsnum = os.path.basename(datapack).split('_')[0][1:]
        s = "{} & {:.1f}% & {:.1f}% & {:.0f} & {:.0f} & {:.1f}% & {:.1f}% & {:.0f} & {:.0f}\\".format(obsnum, *z[1:])
        print(s)
    s = "{} & {:.1f}% & {:.1f}% & {:.0f} & {:.0f} & {:.1f}% & {:.1f}% & {:.0f} & {:.0f}".format("Total",
                                                                                        float(tFNR_nn*100), float(tFPR_nn*100), float(FN_nn.sum()), float(FP_nn.sum()), float(tFNR_reinout*100), float(tFPR_reinout*100), float(FN_reinout.sum()), float(FP_reinout.sum()))
    print(s)

    print(np.mean(sizes),"points per observation")

    tp = np.array(tp_thresh).astype(float).sum(0)
    tn = np.array(tn_thresh).astype(float).sum(0)
    fp = np.array(fp_thresh).astype(float).sum(0)
    fn = np.array(fn_thresh).astype(float).sum(0)
    #S
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    tnr = tn / (tn + fp)
    acc = (tn + tp) / (tp + tn + fp + fn)

    plt.figure(figsize=(5,8))
    plt.plot(fnr*100, fpr*100, c='black')

    argmin = np.argmin(np.abs(fnr) + np.abs(fpr))
    argmin = 28
    plt.scatter(fnr[argmin]*100, fpr[argmin]*100, color='green', label='optimal')
    print("Optimal thresh {}".format(dir_thresh[argmin]))

    print(list(zip(dir_thresh, fpr)))

    plt.xlabel('FNR [%]')
    plt.ylabel('FPR [%]')
    plt.legend()

    plt.savefig('./flag_roc.pdf')
    plt.show()




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
