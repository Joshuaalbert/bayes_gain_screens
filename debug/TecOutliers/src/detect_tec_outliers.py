import tensorflow.compat.v1 as tf
import numpy as np
from bayes_gain_screens.datapack import DataPack
from bayes_gain_screens.plotting import animate_datapack
import os, argparse
import logging

tf.get_logger().setLevel('ERROR')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

def main(datapack, model_dir, version, solset, plot_outliers, batch_size, plot_dir,ncpu):
    """
    This will take the examples in the examples dir and split into train and test sets.
    Train a classifier.
    Saving the model to a cloud linked directory.
    Args:
        examples_dir:
        persistent_space:
        model_dir:

    Returns:

    """
    model_dir = os.path.abspath(os.path.expanduser(model_dir))
    datapack = os.path.abspath(os.path.expanduser(datapack))
    model_path = os.path.join(model_dir,str(version))
    with tf.Session(graph=tf.Graph()) as sess:
        loaded_model = tf.saved_model.load(sess,tags=[tf.saved_model.tag_constants.SERVING],
                                           export_dir=model_path)
        # print(loaded_model.signatures.keys())
        sig_def = loaded_model.signature_def['predict_activity']
        inputs = sig_def.inputs
        outputs = sig_def.outputs
        probs = tf.get_default_graph().get_tensor_by_name(outputs['probability'])
        classification = tf.get_default_graph().get_tensor_by_name(outputs['class'])
        tec = tf.get_default_graph().get_tensor_by_name(inputs['tec'])
        pos = tf.get_default_graph().get_tensor_by_name(inputs['pos'])



        dp = DataPack(datapack,readonly=False)
        dp.current_solset = solset
        dp.select(pol=0)
        tec, axes= dp.tec
        Npol, Nd, Na, Nt = tec.shape
        _, directions = dp.get_directions(axes['dir'])
        directions = np.stack([directions.ra.deg, directions.dec.deg],axis=1)
        inputs = tec[0,...].transpose((1,2,0)).reshape((Na,Nt,Nd,1))/55.

        outputs = []
        if batch_size is None:
            batch_size = Na
        for start in range(0, Na, batch_size):
            stop = min(start + batch_size, Na)
            print("Prediction out batch {}".format(slice(start, stop)))
            output = sess.run(classification, dict(tec=inputs[start:stop,:,:,:], pos=directions))
            detection = output.astype(np.bool)#Na,Nt,Nd,1
            outputs.append(detection)

    outputs = np.concatenate(outputs, axis=0)
    detection = outputs.transpose((3, 2, 0, 1))  # 1,Nd,Na,Nt
    tec_uncert, _ = dp.weights_tec
    tec_uncert = np.where(detection, np.inf, tec_uncert)
    dp.weights_tec = tec_uncert

    if plot_outliers:
        if ncpu is None:
            try:
                ncpu = os.cpu_count()
                if 'sched_getaffinity' in dir(os):
                    ncpu = len(os.sched_getaffinity(0))
            except:
                import multiprocessing
                ncpu = multiprocessing.cpu_count()

        animate_datapack(datapack, os.path.abspath(os.path.expanduser(plot_dir)),
                         num_processes=ncpu,
                         solset=solset,
                         observable='tec',
                         vmin=-60., vmax=60.,
                         labels_in_radec=True, plot_crosses=False,
                         phase_wrap=False,
                         overlay_solset=solset)

def add_args(parser):
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument('--batch_size', help='Number of epochs to train for', default=None, type=int, required=False)
    parser.add_argument('--model_dir', help='Model save directory.', default='models', type=str, required=False)
    parser.add_argument('--plot_outliers', help='Whether to plot outliers.', default=True, type="bool", required=False)
    parser.add_argument('--plot_dir', help='Where to plot outliers.', default='./outliers_plot', type=str, required=False)
    parser.add_argument('--ncpu', help='num cpu to plot with.', default=None, type=int, required=False)
    parser.add_argument('--datapack', help='H5parm to apply to.', default='training', type=str,
                        required=False)
    parser.add_argument('--version', help='Version of model to save as.', default=1, type=int, required=False)
    parser.add_argument('--solset', help='Which solset to get tec from.', default='directionally_referenced', type=str, required=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Tec outlier detection.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_args(parser)
    flags, unparsed = parser.parse_known_args()
    print("Running with:")
    for option, value in vars(flags).items():
        print("\t{} -> {}".format(option, value))
    main(**vars(flags))