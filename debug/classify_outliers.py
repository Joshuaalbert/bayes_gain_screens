from bayes_gain_screens.outlier_detection import remove_outliers
import argparse, os, glob


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
    parser.add_argument('--eval_dir', help='Which model dir to get eval model from. If none, uses working_dir/models.',
                        default=None, type="str_or_none", required=False)
    parser.add_argument('--do_click', help='whether to click.',
                        default=False, type='bool', required=False)
    parser.add_argument('--do_train', help='whether to train.',
                        default=False, type='bool', required=False)
    parser.add_argument('--do_eval', help='whether to eval.',
                        default=False, type='bool', required=False)
    parser.add_argument('--datapack_pattern', help='Pattern of datapacks.',
                        default=None, type=str, required=False)
    parser.add_argument('--ref_image_pattern', help='Pattern of ref_images.',
                        default=None, type=str, required=False)

    parser.add_argument('--L', help='Layers of CNN',
                        default=5, type=int, required=False)
    parser.add_argument('--K', help='Number of nearest neibours to train',
                        default=7, type=int, required=False)
    parser.add_argument('--n_features', help='Number of features to train',
                        default=24, type=int, required=False)
    parser.add_argument('--crop_size', help='In training, length of minibatch',
                        default=250, type=int, required=False)
    parser.add_argument('--batch_size', help='Size of batches to train',
                        default=16, type=int, required=False)
    parser.add_argument('--epochs', help='Number of whole batches to train',
                        default=30, type=int, required=False)

def main(do_click, do_train, do_eval, working_dir, eval_dir, datapack_pattern, ref_image_pattern,
         L,K,n_features, crop_size, batch_size, epochs):
    if do_click or do_train:
        if working_dir is None:
            raise ValueError("working dir can't be none if clicking or training")
        working_dir = os.path.abspath(working_dir)
        os.makedirs(working_dir, exist_ok=True)
    if do_click or do_eval:
        if datapack_pattern is None:
            raise ValueError("datapack pattern can't be none if clicking")
        if ref_image_pattern is None:
            raise ValueError("ref image pattern can't be none in clicking")
        datapacks = glob.glob(datapack_pattern)
        ref_images = glob.glob(ref_image_pattern)
        if len(ref_images) == 1 and len(ref_images) != len(datapacks):
            ref_images = ref_images*len(datapacks)
    else:
        datapacks = None
        ref_images = None

    remove_outliers(do_clicking=do_click,
                    do_training=do_train,
                    do_evaluation=do_eval,
                    datapacks=datapacks,
                    ref_images=ref_images,
                    working_dir=working_dir,
                    eval_dir=eval_dir,
                    L=L, K=K, n_features=n_features, crop_size=crop_size, batch_size=batch_size, epochs=epochs)

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
