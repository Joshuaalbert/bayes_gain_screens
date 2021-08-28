import numpy as np
import pylab as plt
import logging

from bayes_gain_screens.nn_tools import vanilla_training_loop, TrainOneEpoch, AbstractModule

import sonnet as snt
import argparse
import os
import sys
import astropy.units as au

from h5parm import DataPack

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

logger = logging.getLogger(__name__)
import tensorflow as tf

def tf_generator_training_data(time, freqs):
    time = tf.convert_to_tensor(time, dtype=tf.float32)
    freqs = tf.convert_to_tensor(freqs, dtype=tf.float32)
    dt = tf.math.abs(tf.reduce_mean(time[1:] - time[:-1]))
    T = time.shape[0]
    Nf = freqs.shape[0]

    tec_conv = -8.4479745e6 / freqs  # mTECU/Hz
    clock_conv = (2. * np.pi * 1e-9 * freqs)

    @tf.function
    def _single_sample():
        const_amp = tf.random.uniform((), minval=0., maxval=np.pi)
        const_phase = tf.random.uniform((), minval=0., maxval=np.pi)
        const = const_amp * tf.cos(2. * np.pi * time / (2 * 3600.) + const_phase)

        clock_amp = tf.random.uniform((), minval=0., maxval=0.1)
        clock_phase = tf.random.uniform((), minval=0., maxval=np.pi)
        clock = clock_amp * tf.cos(2. * np.pi * time / (2 * 3600.) + clock_phase)

        tec = 0.

        tec_amp = tf.random.uniform((), minval=0., maxval=300.)
        tec_phase = tf.random.uniform((), minval=0., maxval=np.pi)
        tec_period = tf.random.uniform((), minval=dt * 100, maxval=dt * 200)
        tec = tec + tec_amp * (tf.cos(2. * np.pi * time / tec_period + tec_phase) ** 4 - 0.5)

        tec_amp = tf.random.uniform((), minval=0., maxval=30.)
        tec_phase = tf.random.uniform((), minval=0., maxval=np.pi)
        tec_period = tf.random.uniform((), minval=dt * 30, maxval=dt * 50)
        tec = tec + tec_amp * (tf.cos(2. * np.pi * time / tec_period + tec_phase) ** 3 - 0.5)

        tec_amp = tf.random.uniform((), minval=0., maxval=5.)
        tec_phase = tf.random.uniform((), minval=0., maxval=np.pi)
        tec_period = tf.random.uniform((), minval=dt, maxval=dt * 3)
        tec = tec + tec_amp * (tf.cos(2. * np.pi * time / tec_period + tec_phase) ** 4 - 0.5)

        time_offset = tf.random.uniform((), minval=tf.reduce_min(time), maxval=tf.reduce_max(time))
        flip = tf.random.uniform(()) < 0.5

        tec = tec * (
                (-tf.tanh(tf.where(flip, -1., 1.) * (time - time_offset) / (0.25 * T) * 10) + 1.1) / 2.)  # ** (0.5)

        phase = tec[:, None] * tec_conv + clock[:, None] * clock_conv + const[:, None]

        outliers = tf.zeros(phase.shape, tf.bool)
        offset = tf.zeros(phase.shape)

        def body(state):
            (i, outliers, offset) = state
            channel = tf.random.uniform((), minval=0, maxval=outliers.shape[1] - 1, dtype=tf.int64)
            start = tf.random.uniform((), minval=0, maxval=outliers.shape[0], dtype=tf.int64)
            stop = tf.minimum(start + tf.random.uniform((), minval=50, maxval=250, dtype=tf.int64), outliers.shape[0])
            time_mask = (tf.range(outliers.shape[0], dtype=tf.int64) >= start) & (
                    tf.range(outliers.shape[0], dtype=tf.int64) < stop)  # T

            do_side_by_side = tf.random.uniform(()) < 0.5
            do_pert = tf.random.uniform(()) < 0.5

            chan_mask = (tf.range(outliers.shape[1], dtype=tf.int64) == channel)
            mask = (time_mask[:, None] & chan_mask[None, :])
            outliers = mask | outliers
            offset = tf.where(mask,
                              tf.where(do_pert,
                                       tf.random.uniform((), maxval=np.pi * 2.),
                                       np.pi * tf.random.normal(outliers.shape)),
                              offset
                              )

            chan_mask = (tf.range(outliers.shape[1], dtype=tf.int64) == channel + 1)
            mask = (time_mask[:, None] & chan_mask[None, :]) & do_side_by_side
            outliers = mask | outliers
            offset = tf.where(mask,
                              tf.where(do_pert,
                                       tf.random.uniform((), maxval=np.pi * 2.),
                                       np.pi * tf.random.normal(outliers.shape)),
                              offset
                              )

            return (i + 1, outliers, offset)

        state = (np.asarray(0), outliers, offset)
        while state[0] < 20:
            state = body(state)
        (_, outliers, offset) = state

        phase = tf.where(outliers, phase + offset, phase)

        amp_amp = tf.random.uniform((), minval=0.1, maxval=0.5)
        amp_phase = tf.random.uniform((), minval=0., maxval=np.pi)
        amp_time = 1. + amp_amp * tf.cos(2. * np.pi * time / (2 * 3600.) + amp_phase)
        amp_freq = 1. + amp_amp * tf.cos(2. * np.pi * freqs / (freqs[-1] - freqs[0]) + amp_phase)
        amp = amp_time[:, None] * amp_freq[None, :]
        Y = tf.concat([amp * tf.cos(phase), amp * tf.sin(phase)], axis=-1)

        uncert_amp = tf.random.uniform((), minval=0.01, maxval=1.25)
        uncert = uncert_amp * tf.random.normal(Y.shape) * ((-tf.tanh(
            (time[:, None] - tf.random.uniform(()) * tf.reduce_max(time)) / tf.sqrt(
                tf.reduce_mean((time - tf.reduce_mean(time)) ** 2)) * 10) + 1.3) / 2.) ** (0.5)

        Y_obs = Y + uncert

        phase = tf.atan2(Y_obs[:, Nf:], Y_obs[:, :Nf])

        Y_obs = tf.stack([Y_obs[:, :Nf], Y_obs[:, Nf:]], axis=-1)
        Y = tf.stack([Y[:, :Nf], Y[:, Nf:]], axis=-1)
        return tec, phase, Y, Y_obs, tf.cast(outliers, tf.int64)

    return _single_sample


def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))


class ResidualBlock(AbstractModule):
    def __init__(self, N, name=None):
        super(ResidualBlock, self).__init__(name=name)

        self.conv_layers = snt.Sequential([
            snt.Conv2D(N*2, (3, 5), padding='SAME',name='conv_1'), tf.nn.relu,
            snt.Conv2D(N, (3, 3), padding='SAME',name='conv_2'), tf.nn.relu,
            snt.Conv2D(N, (3, 3), padding='SAME',name='conv_3'), tf.nn.relu,
            snt.Conv2D(N*2, (3, 5), padding='SAME',name='conv_4')])

        self.pre_output = snt.Conv2D(N*2, 1, padding='SAME',name='conv_5')

    def _build(self, img, **kwargs):
        img = self.conv_layers(img) + self.pre_output(img)
        return img


class Model(AbstractModule):
    def __init__(self, name=None):
        super(Model, self).__init__(name=name)

        self.res_layers = snt.Sequential([ResidualBlock(8,name='res_1'),
                                           ResidualBlock(16,name='res_2'),
                                           snt.Conv2D(3, 1, padding='SAME', name='conv')])

    def _build(self, batch, **kwargs):
        (img, img_true, outliers) = batch
        del outliers
        del img_true
        img = tf.concat([img, tf.zeros_like(img[..., 0:1])], axis=-1)
        logits = self.res_layers(img)
        return logits


def make_dataset(freqs, times):
    num_examples = 8 * 500
    if freqs is None:
        freqs = np.linspace(121e6, 166e6, 24)
    if times is None:
        times = np.linspace(0., 1039 * 30, 1040)
    _single_sample = tf_generator_training_data(times, freqs)
    dataset = tf.data.Dataset.from_tensor_slices(tf.range(num_examples)).map(lambda i: _single_sample(),
                                                                             deterministic=False)
    dataset = dataset.map(lambda tec, phase, Y, Y_obs, outliers: (Y_obs, Y, outliers))
    return dataset


def main(data_dir, obs_num, working_dir, plot_results):
    dds4_h5parm = os.path.join(data_dir, 'L{}_DDS4_full_merged.h5'.format(obs_num))
    logger.info(f"Performing outlier detection on gains from {dds4_h5parm}.")

    plot_dir = os.path.join(working_dir, 'plots')
    if plot_results:
        os.makedirs(plot_dir, exist_ok=True)

    with DataPack(dds4_h5parm, readonly=True) as dp:
        dp.select(ant=None, dir=None, time=None, freq=None, pol=slice(0, 1, 1))
        axes = dp.axes_phase
        _, freqs = dp.get_freqs(axes['freq'])
        _, times = dp.get_times(axes['time'])
        times = times.mjd
        times -= times[0]
        times *= 86400
        freqs = freqs.to(au.Hz).value

    logger.info("Learning the neural flagger model.")
    training_dataset = make_dataset(freqs, times).batch(8)
    test_dataset = make_dataset(freqs, times).batch(8)

    def loss(model_outputs, batch):
        pred_img = model_outputs[..., 0:2]
        pred_logits = model_outputs[..., 2:3]
        (img, img_true, outliers) = batch
        _outliers = tf.cast(outliers, img_true.dtype)
        rec_loss = tf.reduce_sum(tf.square(pred_img - img_true) * _outliers[..., None]) / (
                tf.reduce_sum(_outliers) + 0.00001)
        class_loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(outliers, pred_logits[..., 0], from_logits=True))
        return rec_loss + class_loss

    opt = snt.optimizers.Adam(1e-3, beta1=1 - 1 / 10, beta2=1 - 1 / 50)
    model = Model()
    train_one_epoch = TrainOneEpoch(model, loss, opt)
    vanilla_training_loop(train_one_epoch, training_dataset=training_dataset, test_dataset=test_dataset,
                          num_epochs=1000, early_stop_patience=3,
                          log_dir=os.path.join(working_dir, 'tf_logs'),
                          checkpoint_dir=os.path.join(working_dir, 'tf_checkpoints'))

    @tf.function
    def predict(batch):
        return tf.nn.sigmoid(model(batch)[..., 2:3])

    # calibrate threshold
    logger.info("Determining optimal threshold from ROC curve")

    @tf.function
    def calc_fpr_fnr(thresholds):
        tp, fp, tn, fn = [tf.zeros((thresholds.shape[0], 24))] * 4
        for batch in iter(test_dataset):
            outlier_prob = predict(batch)[:, :, :, 0]
            outliers = tf.cast(batch[2], tf.bool)
            c = tf.cast(outlier_prob, thresholds.dtype) > thresholds[:, None, None, None]
            tp += tf.reduce_sum(tf.cast(outliers & c, tp.dtype), axis=[1, 2])
            tn += tf.reduce_sum(tf.cast((~outliers) & (~c), tp.dtype), axis=[1, 2])
            fp += tf.reduce_sum(tf.cast((~outliers) & c, tp.dtype), axis=[1, 2])
            fn += tf.reduce_sum(tf.cast(outliers & (~c), tp.dtype), axis=[1, 2])
        fpr = fp / (fp + tn)
        fnr = fn / (fn + tp)
        return fpr, fnr

    thresholds = np.linspace(0., 1., 100)
    fpr, fnr = calc_fpr_fnr(thresholds)
    fpr = fpr.numpy()
    fnr = fnr.numpy()
    optimal_idx = np.argmin(np.abs(fpr) + np.abs(fnr), axis=0)
    optimal_thresholds = thresholds[optimal_idx]
    optimal_fpr = np.take_along_axis(fpr, optimal_idx[None,:], axis=0)[0,:]
    optimal_fnr = np.take_along_axis(fnr, optimal_idx[None,:], axis=0)[0,:]

    with DataPack(dds4_h5parm, readonly=True) as dp:
        dp.select(ant=None, dir=None, time=None, freq=None, pol=slice(0, 1, 1))
        phase, axes = dp.phase
        amp, axes = dp.amplitude
        _, Nd, Na, Nf, Nt = amp.shape
        antennas, _ = dp.get_antennas(axes['ant'])
        patch_names, _ = dp.get_directions(axes['dir'])
        Y_obs = np.stack([amp * np.cos(phase), amp * np.sin(phase)], axis=-1)
        Y_obs = Y_obs.reshape((Nd * Na, Nf, Nt, 2)).transpose((0, 2, 1, 3)).astype(np.float32)

    for i, (freq, _fpr, _fnr, threshold) in enumerate(zip(freqs, optimal_fpr, optimal_fnr, optimal_thresholds)):
        logger.info(
            "Freq : {:.1f} MHz | threshold = {:.3f} | FPR = {:.3f} | FNR = {:.3f}".format(freq / 1e6, threshold, _fpr,
                                                                                          _fnr))
        if plot_results:
            plt.plot(fpr[:, i], fnr[:, i])
            plt.title(
                "Freq : {:.1f} MHz | threshold = {:.3f} | FPR = {:.3f} | FNR = {:.3f}".format(freq / 1e6, threshold,
                                                                                              _fpr, _fnr))
            plt.scatter(_fpr, _fnr, c='red')
            plt.savefig(os.path.join(plot_dir, 'roc_{:.1f}MHz.png'.format(freq / 1e6)))
            plt.close('all')

    dataset = tf.data.Dataset.from_tensor_slices(Y_obs).batch(1)
    outliers_detected = []
    for i, Y_obs in enumerate(iter(dataset)):
        (dir_idx, ant_idx) = np.unravel_index(i, (Nd, Na))
        logits = predict((Y_obs, None, None))
        outliers_detected.append(logits.numpy()[:, :, :, 0] > optimal_thresholds)
        phase_obs = np.arctan2(Y_obs[..., 1].numpy(), Y_obs[..., 0].numpy())
        fig, axs = plt.subplots(2, 1, sharey=True, sharex=True)
        axs[0].imshow(phase_obs[0].T, aspect='auto', interpolation='nearest', cmap='twilight', vmin=-np.pi, vmax=np.pi)
        axs[1].imshow(outliers_detected[-1][0].T, aspect='auto', interpolation='nearest')
        fig.savefig(os.path.join(plot_dir, 'outliers_dir{:02d}_ant{:02d}.png'.format(dir_idx, ant_idx)))
        plt.close('all')

    outliers_detected = np.concatenate(outliers_detected, axis=0)
    for i in range(Nf):
        logger.info("Outliers freq : {:.1f}MHz : {} / {} ({}%)".format(freqs[i] / 1e6,
                                                                       outliers_detected[...,i].sum(0).sum(0),
                                                                       Nd * Na * Nt,
                                                                       100 * outliers_detected[...,i].sum(0).sum(0) / (Nd * Na * Nt)))
    outliers_detected = outliers_detected.transpose((0, 2, 1)).reshape((Nd, Na, Nf, Nt))
    with DataPack(dds4_h5parm, readonly=False) as dp:
        dp.select(ant=None, dir=None, time=None, freq=None, pol=slice(0, 1, 1))
        dp.weights_phase = outliers_detected[None, ...]


def debug_main():
    main(obs_num=342938,
         data_dir="/home/albert/data/gains_screen/data",
         working_dir="/home/albert/data/gains_screen/data",
         plot_results=True)


def add_args(parser):
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument('--obs_num', help='Obs number L*',
                        default=None, type=int, required=True)
    parser.add_argument('--data_dir', help='Where are the data files are stored.',
                        default=None, type=str, required=True)
    parser.add_argument('--working_dir', help='Where to perform the computation.',
                        default=None, type=str, required=True)
    parser.add_argument('--plot_results', help='Whether to plot results.',
                        default=True, type="bool", required=False)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        debug_main()
        exit(0)
    parser = argparse.ArgumentParser(
        description='Trains a neural network to detect outliers, and then applies it to the gains stored in an h5parm.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_args(parser)
    flags, unparsed = parser.parse_known_args()
    logger.info("Running with:")
    for option, value in vars(flags).items():
        logger.info("    {} -> {}".format(option, value))
    main(**vars(flags))