import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
import numpy as np
from bayes_gain_screens.datapack import DataPack

import logging

import argparse, sys, glob, os


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
####
# D_i ~ ScreenType(D_i) = Dirichlet(alpha)
# X_i ~ Screen(X_i | D_i) = Gaussian
# P(D_i, X_i) = ScreenType(D_i)Screen(X_i | D_i)
#

class Encoder(tf.keras.layers.Layer):
    def __init__(self, layer_size=128, latent_size=8, rate=0.1, name=None):
        super(Encoder, self).__init__(name=name)
        self.proj1 = tf.keras.layers.Dense(layer_size)
        self.d1 = tf.keras.layers.Dense(layer_size, activation=tf.nn.relu)
        self.d2 = tf.keras.layers.Dense(layer_size, activation=tf.nn.relu)
        self.do1 = tf.keras.layers.Dropout(rate)
        self.do2 = tf.keras.layers.Dropout(rate)
        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.proj2 = tf.keras.layers.Dense(latent_size)

    def call(self, inputs, training):
        outputs = self.proj1(inputs)
        outputs = self.ln1(self.do1(self.d1(inputs), training=training) + outputs)
        outputs = self.ln2(self.do2(self.d2(inputs), training=training) + outputs)
        outputs = self.proj2(outputs)
        return outputs

class Decoder(tf.keras.layers.Layer):
    def __init__(self, feature_size, layer_size=128, rate=0.1, name=None):
        super(Decoder, self).__init__(name=name)
        self.proj1 = tf.keras.layers.Dense(layer_size)
        self.d1 = tf.keras.layers.Dense(layer_size, activation=tf.nn.relu)
        self.d2 = tf.keras.layers.Dense(layer_size, activation=tf.nn.relu)
        self.do1 = tf.keras.layers.Dropout(rate)
        self.do2 = tf.keras.layers.Dropout(rate)
        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.proj2 = tf.keras.layers.Dense(feature_size)

    def call(self, inputs, training):
        outputs = self.proj1(inputs)
        outputs = self.ln1(self.do1(self.d1(inputs), training=training) + outputs)
        outputs = self.ln2(self.do2(self.d2(inputs), training=training) + outputs)
        outputs = self.proj2(outputs)
        return outputs

class AutoEncoderDirichlet(tf.keras.Model):
    def __init__(self, layer_size, feature_size, num_classes, rate=0.1, S=1, name=None):
        super(AutoEncoderDirichlet, self).__init__(name=name)
        self.S = S
        self.F = feature_size
        self.encoder = Encoder(layer_size=layer_size,latent_size=num_classes, rate=rate)
        self.decoder = Decoder(layer_size=layer_size,feature_size=feature_size, rate=rate)
        self.alpha = tf.Variable(tf.random.uniform([num_classes]), trainable=True)
        self.prior = tfp.distributions.Dirichlet(0.01+tf.nn.relu(self.alpha))
        self.fake_class_input = tf.one_hot(tf.range(num_classes, dtype=tf.int32)[None, :],num_classes,dtype=tf.float32)

    def call(self, inputs, training=None, mask=None):
        #B, C
        variational_alpha = 0.01+tf.nn.relu(self.encoder(inputs, training=training))
        variational_posterior = tfp.distributions.Dirichlet(variational_alpha)
        posterior_mean = variational_posterior.mean()
        KL = variational_posterior.kl_divergence(self.prior)
        #S,B,C
        posterior_samples = variational_posterior.sample(self.S)
        #S, B, F
        model_data = self.decoder(posterior_samples, training=training)
        likelihood = tfp.distributions.Bernoulli(logits=model_data, dtype=tf.float32)
        # likelihood = tfp.distributions.MultivariateNormalDiag(loc=model_data[...,:self.F], scale_diag=tf.nn.softplus(model_data[...,self.F:]))
        var_exp = tf.reduce_mean(tf.reduce_sum(likelihood.log_prob(inputs),axis=-1), axis=0)
        # with tf.control_dependencies([tf.print(['KL', tf.reduce_mean(KL), 'var_exp', tf.reduce_mean(var_exp)])]):
        elbo = var_exp - KL

        #B, 28*28
        fake_model_data = self.decoder(self.fake_class_input, training=False)
        fake_model_data = tf.reshape(fake_model_data, (-1, 28,28, 1))
        return tf.reduce_mean(tf.negative(elbo)), KL, var_exp, posterior_samples, fake_model_data, posterior_mean

class AutoEncoderGaussian(tf.keras.Model):
    def __init__(self, layer_size, feature_size, latent_size, rate=0.1, S=1, name=None):
        super(AutoEncoderGaussian, self).__init__(name=name)
        self.S = S
        self.F = feature_size
        self.L = latent_size
        self.encoder = Encoder(layer_size=layer_size,latent_size=latent_size*2, rate=rate)
        self.decoder = Decoder(layer_size=layer_size,feature_size=feature_size*2, rate=rate)
        self.prior = tfp.distributions.MultivariateNormalDiag(loc=tf.zeros((latent_size,)),scale_identity_multiplier=1.)
        xx,yy = tf.meshgrid(tf.linspace(-2.,2.,5),tf.linspace(-2.,2.,5), indexing='ij')
        self.fake_input = tf.stack([tf.reshape(xx,(-1,)), tf.reshape(yy,(-1,))], axis=-1)[None,...]

    def call(self, inputs, training=None, mask=None):
        #B, L*2
        variational_params = self.encoder(inputs, training=training)
        variational_posterior = tfp.distributions.MultivariateNormalDiag(variational_params[...,:self.L], scale_diag=tf.math.exp(variational_params[...,self.L:]))
        posterior_mean = variational_posterior.mean()
        KL = variational_posterior.kl_divergence(self.prior)
        #S,B,C
        posterior_samples = variational_posterior.sample(self.S)
        #S, B, F
        model_data = self.decoder(posterior_samples, training=training)
        # likelihood = tfp.distributions.Bernoulli(logits=model_data, dtype=tf.float32)
        likelihood = tfp.distributions.MultivariateNormalDiag(loc=model_data[...,:self.F], scale_diag=tf.math.exp(model_data[...,self.F:]))
        var_exp = tf.reduce_mean(tf.reduce_sum(likelihood.log_prob(inputs),axis=-1), axis=0)
        # with tf.control_dependencies([tf.print(['KL', tf.reduce_mean(KL), 'var_exp', tf.reduce_mean(var_exp)])]):
        elbo = var_exp - KL

        #B, 28*28
        fake_model_data = self.decoder(self.fake_input, training=False)
        # fake_likelihood = tfp.distributions.MultivariateNormalDiag(loc=model_data[...,:self.F], scale_diag=tf.math.exp(model_data[...,self.F:]))

        # fake_model_data = tf.reshape(fake_model_data, (-1, 28,28, 1))
        return tf.reduce_mean(tf.negative(elbo)), KL, var_exp, posterior_samples, model_data[...,:self.F], posterior_mean

class MnistExampleGenerator:
    def __init__(self):
        (self.x_train, _), (self.x_test, _) = tf.keras.datasets.mnist.load_data()

    def __call__(self, train=True):
        if train:
            for i in range(self.x_train.shape[0]):
                yield (self.x_train[i, :, :].flatten()/255.,)
            else:
                for i in range(self.x_test.shape[0]):
                    yield (self.x_test[i, :, :].flatten()/255.,)

class ExampleGenerator:
    def __init__(self):
        sols = glob.glob("/home/albert/git/bayes_gain_screens/notebooks/chap4/images/biased/*.h5")
        tecs = []
        for sol in sols:
            dp = DataPack(sol,readonly=True)
            dp.current_solset = 'screen_posterior'
            dp.select(pol=0, ant=slice(1,None,None))
            tec, axes = dp.tec
            Npol, Nd, Na, Nt = tec.shape
            tecs.append(tec.transpose((0,2,3,1)).reshape((-1, Nd)))
        tecs = np.concatenate(tecs,axis=0)
        self.tecs = tecs
        self.Nd = self.tecs.shape[1]

    def __call__(self, train=True):
        if train:
            for i in range(self.tecs.shape[0]):
                yield (self.tecs[i, :].flatten(),)

class Train():
    def __init__(self, layer_size=128, feature_size=28*28, latent_size=8, rate=0.1, S=10, batch_size=64):

        graph = tf.Graph()
        with graph.as_default():
            train_set_pl = tf.placeholder(tf.bool)
            training_pl = tf.placeholder(tf.bool)
            example_gen = ExampleGenerator()
            feature_size = example_gen.Nd
            dataset = tf.data.Dataset.from_generator(
                example_gen,
                output_types=(tf.float32,),
                output_shapes=((feature_size,),),
            args=(train_set_pl,)).shuffle(100000).batch(batch_size=batch_size, drop_remainder=True)
            iterator_tensor = dataset.make_initializable_iterator()
            self.init = iterator_tensor.initializer
            features,  = iterator_tensor.get_next()

            auto_encoder = AutoEncoderGaussian(layer_size=layer_size, feature_size=feature_size, latent_size=latent_size, rate=rate, S=S)
            loss, KL, var_exp, posterior_samples, fake_model_data, posterior_mean= auto_encoder(features, training=False)

            global_step = tf.Variable(0, trainable=False)
            lr = tf.train.inverse_time_decay(1e-2, global_step, 1000, 1.)

            optimiser = tf.train.AdamOptimizer(lr)
            opt_op = optimiser.minimize(loss, global_step=global_step)
            self.train_summaries = tf.summary.merge([tf.summary.scalar('loss', loss, family='train'),
                                                     tf.summary.scalar('learning_rate', lr, family='train'),
                                                     tf.summary.scalar('KL', tf.reduce_mean(KL), family='train'),
                                                     tf.summary.scalar('var_exp', tf.reduce_mean(var_exp), family='train'),
                                                     # tf.summary.image('class_mean', fake_model_data, max_outputs=25,
                                                     #                  family='train')
                                                     ]
                                                    )

            ###
            # eval model code
            eval_features = tf.placeholder(tf.float32, shape=[None, feature_size])
            _, _, _, _, _, posterior_mean = auto_encoder(eval_features, training=False)

        self.lr = lr
        self.opt_op = opt_op
        self.loss = loss
        self.global_step = global_step
        self.train_set = train_set_pl
        self.training_pl = training_pl
        self.eval_features = eval_features
        self.eval_posterior_mean = posterior_mean
        self.graph = graph


    def export_model(self, sess:tf.Session, model_dir, version):
        with sess.graph.as_default():
            export_path = os.path.join(model_dir, 'model', str(version))
            logger.info('Exporting trained model to {}'.format(export_path))
            builder = tf.saved_model.builder.SavedModelBuilder(export_path)
            # the inputs and outputs
            tensor_info_input = tf.saved_model.utils.build_tensor_info(self.eval_features)
            tensor_info_output = tf.saved_model.utils.build_tensor_info(self.eval_posterior_mean)
            # signature made using util
            prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={'features': tensor_info_input},
                    outputs={'probability': tensor_info_output},
                    method_name=tf.saved_model.signature_constants
                        .PREDICT_METHOD_NAME)
            # adds just the required ops to evaluate
            builder.add_meta_graph_and_variables(
                sess, [tf.saved_model.tag_constants.SERVING],
                signature_def_map={
                    'predict_activity':
                        prediction_signature,
                },
                # main_op=tf.tables_initializer(),
                strip_default_attrs=True)

            builder.save()

    def train(self, epochs, training_dir, model_dir, version, print_freq=10):
        os.makedirs(model_dir, exist_ok=True)
        training_dir = os.path.join(training_dir, 'run{}'.format(len(glob.glob(os.path.join(training_dir,'run*')))))
        os.makedirs(training_dir, exist_ok=True)

        logdir = os.path.join(training_dir, 'logs')  # auto generated

        with tf.Session(graph=self.graph) as sess:
            with tf.summary.FileWriter(logdir, sess.graph, session=sess) as writer:
                saver = tf.train.Saver()
                print("Initialising valiables")
                sess.run(tf.initialize_variables(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)))
                print("Restoring if possibe")
                try:
                    saver.restore(sess, tf.train.latest_checkpoint(model_dir))
                except:
                    print("No model found to restore")
                    pass
                for epoch in range(epochs):
                    sess.run([self.init],
                             {self.train_set: True})
                    train_loss = 0.
                    batch = 0
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    while True:
                        try:
                            _, global_step, lr, loss, summaries = sess.run(
                                [self.opt_op, self.global_step, self.lr, self.loss,
                                 self.train_summaries], {self.training_pl:True}
                                )
                            train_loss = train_loss + loss
                            batch += 1
                            if global_step % print_freq == 0:
                                writer.add_summary(summaries, global_step)
                                print("Epoch {:02d} Step {:04d} Train loss {:.5f} Learning rate {:.5f}".format(epoch,
                                                                                                               global_step,
                                                                                                               loss,
                                                                                                               lr))

                        except tf.errors.OutOfRangeError:
                            break


                    print("Train Results: Epoch {:02d} Train loss {:.5f}".format(epoch, train_loss / batch))
                    print('Saving...')
                    save_path = saver.save(sess, os.path.join(training_dir, 'model'), global_step=self.global_step)
                    print("Saved to {}".format(save_path))
            # save model
            self.export_model(sess, model_dir, version)

def main(epochs, batch_size, model_dir, training_dir, version):
    trainer = Train( layer_size=128, feature_size=28*28, latent_size=2, rate=0.1, S=50, batch_size=batch_size)
    trainer.train(epochs, training_dir, model_dir, version, print_freq=10)


def add_args(parser):
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument('--epochs', help='Number of epochs to train for',
                        default=10, type=int, required=False)
    parser.add_argument('--batch-size', help='Number of epochs to train for',
                        default=16, type=int, required=False)
    parser.add_argument('--model-dir', help='Model save directory.',
                        default=None, type=str, required=True)
    parser.add_argument('--training-dir', help='Training and logs directory.',
                        default=None, type=str, required=True)
    parser.add_argument('--version', help='Version of model to save as.',
                        default=1, type=int, required=False)

def test_main():
    main(epochs=1, batch_size=256, training_dir='./training',model_dir='./models', version=1)

if __name__ == '__main__':
    if len(sys.argv) == 1:
        test_main()
        exit(0)
    parser = argparse.ArgumentParser(
        description='Train phase screen auto-encoder and visualise classes.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_args(parser)
    flags, unparsed = parser.parse_known_args()
    print("Running with:")
    for option, value in vars(flags).items():
        print("    {} -> {}".format(option, value))
    main(**vars(flags))

