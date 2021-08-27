from jax import numpy as jnp
import pylab as plt
import logging

from bayes_gain_screens.nn_tools import vanilla_training_loop, TrainOneEpoch, AbstractModule

import sonnet as snt

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

logger = logging.getLogger(__name__)
import tensorflow as tf

def tf_generator_training_data(num_examples, dt, T, freqs):
    time = tf.linspace(0., T * dt, T + 1)
    tec_conv = -8.4479745e6 / freqs  # mTECU/Hz
    clock_conv = (2. * jnp.pi * 1e-9 * freqs)

    @tf.function
    def _single_sample():
        const_amp = tf.random.uniform((), minval=0., maxval=jnp.pi)
        const_phase = tf.random.uniform((), minval=0., maxval=jnp.pi)
        const = const_amp * tf.cos(2. * jnp.pi * time / (2 * 3600.) + const_phase)

        clock_amp = tf.random.uniform((), minval=0., maxval=0.1)
        clock_phase = tf.random.uniform((), minval=0., maxval=jnp.pi)
        clock = clock_amp * tf.cos(2. * jnp.pi * time / (2 * 3600.) + clock_phase)

        tec = 0.

        tec_amp = tf.random.uniform((), minval=0., maxval=300.)
        tec_phase = tf.random.uniform((), minval=0., maxval=jnp.pi)
        tec_period = tf.random.uniform((), minval=dt * 100, maxval=dt * 200)
        tec = tec + tec_amp * (tf.cos(2. * jnp.pi * time / tec_period + tec_phase) ** 4 - 0.5)

        tec_amp = tf.random.uniform((), minval=0., maxval=30.)
        tec_phase = tf.random.uniform((), minval=0., maxval=jnp.pi)
        tec_period = tf.random.uniform((), minval=dt * 30, maxval=dt * 50)
        tec = tec + tec_amp * (tf.cos(2. * jnp.pi * time / tec_period + tec_phase) ** 3 - 0.5)

        tec_amp = tf.random.uniform((), minval=0., maxval=5.)
        tec_phase = tf.random.uniform((), minval=0., maxval=jnp.pi)
        tec_period = tf.random.uniform((), minval=dt, maxval=dt * 3)
        tec = tec + tec_amp * (tf.cos(2. * jnp.pi * time / tec_period + tec_phase) ** 4 - 0.5)


        time_offset = tf.random.uniform((),minval=tf.reduce_min(time), maxval=tf.reduce_max(time))
        flip = tf.random.uniform(()) < 0.5

        tec = tec * ((-tf.tanh(tf.where(flip, -1., 1.) *(time - time_offset) / (0.25*T) * 10) + 1.1) / 2.)  # ** (0.5)

        phase = tec[:, None] * tec_conv + clock[:, None] * clock_conv + const[:, None]

        outliers = tf.zeros(phase.shape, tf.bool)
        offset = tf.zeros(phase.shape)
        def body(state):
            (i, outliers, offset) = state
            channel = tf.random.uniform((), minval=0, maxval=outliers.shape[1] - 1, dtype=tf.int64)
            start = tf.random.uniform((), minval=0, maxval=outliers.shape[0], dtype=tf.int64)
            stop = tf.minimum(start + tf.random.uniform((),minval=50, maxval=250,dtype=tf.int64), outliers.shape[0])
            time_mask = (tf.range(outliers.shape[0],dtype=tf.int64) >= start) & (tf.range(outliers.shape[0],dtype=tf.int64) < stop)#T

            do_side_by_side = tf.random.uniform(()) < 0.5
            do_pert = tf.random.uniform(()) < 0.5

            chan_mask = (tf.range(outliers.shape[1],dtype=tf.int64) == channel)
            mask = (time_mask[:, None] & chan_mask[None,:])
            outliers = mask | outliers
            offset = tf.where(mask,
                              tf.where(do_pert,
                                       tf.random.uniform((), maxval=jnp.pi * 2.),
                                       jnp.pi * tf.random.normal(outliers.shape)),
                              offset
                              )

            chan_mask = (tf.range(outliers.shape[1],dtype=tf.int64) == channel+1)
            mask = (time_mask[:, None] & chan_mask[None, :]) & do_side_by_side
            outliers = mask | outliers
            offset = tf.where(mask,
                              tf.where(do_pert,
                                       tf.random.uniform((), maxval=jnp.pi * 2.),
                                       jnp.pi * tf.random.normal(outliers.shape)),
                              offset
                              )


            return (i+1, outliers, offset)

        state = (jnp.asarray(0),  outliers, offset)
        while state[0] < 20:
            state = body(state)
        (_, outliers, offset) = state

        phase = tf.where(outliers, phase + offset, phase)

        amp_amp = tf.random.uniform((), minval=0.1, maxval=0.5)
        amp_phase = tf.random.uniform((), minval=0., maxval=jnp.pi)
        amp_time = 1. + amp_amp * tf.cos(2. * jnp.pi * time / (2 * 3600.) + amp_phase)
        amp_freq = 1. + amp_amp * tf.cos(2. * jnp.pi * freqs / (freqs[-1] - freqs[0]) + amp_phase)
        amp = amp_time[:, None] * amp_freq[None, :]
        Y = tf.concat([amp * tf.cos(phase), amp * tf.sin(phase)], axis=-1)

        uncert_amp = tf.random.uniform((), minval=0.01, maxval=1.25)
        uncert = uncert_amp * tf.random.normal(Y.shape) * ((-tf.tanh(
            (time[:, None] - tf.random.uniform(()) * tf.reduce_max(time)) / tf.sqrt(tf.reduce_mean((time - tf.reduce_mean(time))**2)) * 10) + 1.3) / 2.) ** (0.5)

        Y_obs = Y + uncert

        phase = tf.atan2(Y_obs[:, freqs.size:], Y_obs[:, :freqs.size])

        Y_obs = tf.stack([Y_obs[:, :freqs.size], Y_obs[:,freqs.size:]], axis=-1)
        Y = tf.stack([Y[:, :freqs.size], Y[:,freqs.size:]], axis=-1)
        return tec, phase, Y, Y_obs, tf.cast(outliers,tf.int64)

    return _single_sample


def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))

class ResidualBlock(AbstractModule):
    def __init__(self,name=None):
        super(ResidualBlock, self).__init__(name=name)

        self.conv_layers = snt.Sequential([
            snt.Conv2D(8, (3,5), padding='SAME'), tf.nn.leaky_relu,
            snt.Conv2D(16, (3,3), padding='SAME'), tf.nn.leaky_relu,
            snt.Conv2D(3, (3,5), padding='SAME')])

    def _build(self, img, **kwargs):
        img = self.conv_layers(img) + img
        return img

class Model(AbstractModule):
    def __init__(self, name=None):
        super(Model, self).__init__(name=name)

        self.res_layers1 = snt.Sequential([ResidualBlock(), ResidualBlock()])

    def _build(self, batch, **kwargs):
        (img, img_true, outliers) = batch
        del outliers
        del img_true
        img = tf.concat([img, tf.zeros_like(img[..., 0:1])],axis=-1)
        logits1 = self.res_layers1(img)
        return logits1

def make_dataset():
    freqs = jnp.linspace(121e6, 166e6, 24)
    _single_sample = tf_generator_training_data(4 * 100, 30, 1001, freqs)
    dataset = tf.data.Dataset.from_tensor_slices(tf.range(100*4)).map(lambda i: _single_sample(),deterministic=False)
    dataset = dataset.map(lambda tec, phase, Y,  Y_obs, outliers: (Y_obs, Y, outliers))
    return dataset



def main():
    training_dataset = make_dataset().batch(8)
    test_dataset = make_dataset().batch(8)

    def loss(model_outputs, batch):
        pred_img = model_outputs[...,0:2]
        pred_logits = model_outputs[...,2:3]
        (img, img_true, outliers) = batch
        _outliers = tf.cast(outliers, img_true.dtype)
        rec_loss = tf.reduce_sum(tf.square(pred_img - img_true)*_outliers[...,None])/(tf.reduce_sum(_outliers)+0.00001)
        class_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(outliers, pred_logits[...,0], from_logits=True))
        return rec_loss+class_loss

    opt = snt.optimizers.Adam(1e-3,beta1=1-1/10, beta2=1-1/50)
    model = Model()
    train_one_epoch = TrainOneEpoch(model, loss, opt)
    vanilla_training_loop(train_one_epoch,training_dataset=training_dataset,test_dataset=test_dataset,
                          num_epochs=0,early_stop_patience=3,
                          log_dir='tf_logs/04', checkpoint_dir='tf_checkpoints/04')

    @tf.function
    def predict(batch):
        return model(batch)[...,0:2], tf.nn.sigmoid(model(batch)[...,2:3])




    # calibrate

    @tf.function
    def calc_fpr_fnr(thresholds):
        tp,fp,tn,fn=[tf.zeros((thresholds.shape[0], 24))]*4
        for batch in iter(test_dataset):
            outlier_prob = predict(batch)[1][:,:,:,0]
            outliers = tf.cast(batch[2], tf.bool)
            c = tf.cast(outlier_prob, thresholds.dtype) > thresholds[:,None,None,None]
            tp += tf.reduce_sum(tf.cast(outliers & c, tp.dtype), axis=[1,2])
            tn += tf.reduce_sum(tf.cast((~outliers) & (~c), tp.dtype), axis=[1,2])
            fp += tf.reduce_sum(tf.cast((~outliers) & c, tp.dtype), axis=[1,2])
            fn += tf.reduce_sum(tf.cast(outliers & (~c), tp.dtype), axis=[1,2])
        fpr = fp / (fp + tn)
        fnr = fn / (fn + tp)
        return fpr, fnr

    thresholds = jnp.linspace(0., 1., 100)
    fpr, fnr = calc_fpr_fnr(thresholds)
    optimal_idx = jnp.argmin(jnp.abs(fpr.numpy()) + jnp.abs(fnr.numpy()), axis=0)
    print(thresholds[optimal_idx])
    # plt.plot(fpr, fnr)
    # plt.title(thresholds[optimal_idx])
    # plt.scatter(fpr[optimal_idx], fnr[optimal_idx], c='red')
    # plt.show()

    optimal_threshold = thresholds[optimal_idx]

    # for batch in iter(test_dataset):
    #     outlier_prob = predict(batch).numpy() > optimal_threshold
    #     outliers = batch[1].numpy()
    #     Y_obs = batch[0].numpy()
    #     Y_obs = jnp.concatenate([Y_obs[...,0], Y_obs[...,1]],axis=-1)
    #     plt.imshow(Y_obs[0].T, aspect='auto', interpolation='nearest',cmap='PuOr')
    #     plt.show()
    #     plt.imshow(outliers[0].T, aspect='auto', interpolation='nearest')
    #     plt.show()
    #     plt.imshow(outlier_prob[0][:,:,0].T, aspect='auto', interpolation='nearest')
    #     plt.show()


    from h5parm import DataPack
    with DataPack('/home/albert/data/edgecases/L521522_DDS4_full_merged.h5', readonly=True) as dp:
    # with DataPack('/home/albert/data/gains_screen/data/L342938_DDS4_full_merged.h5', readonly=True) as dp:
        dp.select(ant=None, dir=[0,1,10,20,30,40], time=None, freq=None,pol=slice(0,1,1))
        phase, axes = dp.phase
        amp, axes = dp.amplitude
        _, Nd, Na, Nf, Nt = amp.shape
        _, freqs = dp.get_freqs(axes['freq'])
        Y_obs = jnp.stack([amp*jnp.cos(phase), amp*jnp.sin(phase)], axis=-1)
        Y_obs = Y_obs.reshape((Nd*Na, Nf, Nt, 2)).transpose((0,2,1,3)).astype(jnp.float32)

    dataset = tf.data.Dataset.from_tensor_slices(Y_obs).batch(1)
    for i, Y_obs in enumerate(iter(dataset)):
        pred_img, logits = predict((Y_obs, None, None))
        outlier_prob = logits.numpy() > optimal_threshold[:, None]
        # Y_obs = jnp.concatenate([Y_obs[...,0].numpy(), Y_obs[...,1].numpy()],axis=-1)
        phase_obs = jnp.arctan2(Y_obs[...,1].numpy(),Y_obs[...,0].numpy())
        phase_pred = jnp.arctan2(pred_img[...,1].numpy(),pred_img[...,0].numpy())
        fig, axs = plt.subplots(3,1,sharey=True, sharex=True)
        axs[0].imshow(phase_obs[0].T, aspect='auto', interpolation='nearest',cmap='twilight', vmin=-jnp.pi,vmax=jnp.pi)
        axs[1].imshow(phase_pred[0].T, aspect='auto', interpolation='nearest',cmap='twilight', vmin=-jnp.pi,vmax=jnp.pi)
        axs[2].imshow(outlier_prob[0][:,:,0].T, aspect='auto', interpolation='nearest')
        fig.savefig(f'figs/fig{i}.png')
        plt.show()



if __name__ == '__main__':
    main()
