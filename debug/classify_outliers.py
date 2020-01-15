import numpy as np
from bayes_gain_screens.outlier_detection import filter_tec_dir
import glob, os
from bayes_gain_screens.datapack import DataPack
from bayes_gain_screens.misc import voronoi_finite_polygons_2d, get_coordinates
import matplotlib

matplotlib.use('tkagg')
import pylab as plt
from scipy.spatial import Voronoi, cKDTree
from scipy.optimize import linprog
from astropy.io import fits
from astropy.wcs import WCS
import tensorflow.compat.v1 as tf
import networkx as nx
from graph_nets.utils_np import networkxs_to_graphs_tuple
from sklearn.metrics import roc_curve


# from sklearn.model_selection import StratifiedShuffleSplit

def flatten(f):
    """ Flatten a fits file so that it becomes a 2D image. Return new header and data """

    naxis = f[0].header['NAXIS']
    if naxis < 2:
        raise ValueError('Cannot make map from this')
    if naxis == 2:
        return fits.PrimaryHDU(header=f[0].header, data=f[0].data)

    w = WCS(f[0].header)
    wn = WCS(naxis=2)

    wn.wcs.crpix[0] = w.wcs.crpix[0]
    wn.wcs.crpix[1] = w.wcs.crpix[1]
    wn.wcs.cdelt = w.wcs.cdelt[0:2]
    wn.wcs.crval = w.wcs.crval[0:2]
    wn.wcs.ctype[0] = w.wcs.ctype[0]
    wn.wcs.ctype[1] = w.wcs.ctype[1]

    header = wn.to_header()
    header["NAXIS"] = 2
    copy = ('EQUINOX', 'EPOCH', 'BMAJ', 'BMIN', 'BPA', 'RESTFRQ', 'TELESCOP', 'OBSERVER')
    for k in copy:
        r = f[0].header.get(k)
        if r is not None:
            header[k] = r

    slice = []
    for i in range(naxis, 0, -1):
        if i <= 2:
            slice.append(np.s_[:], )
        else:
            slice.append(0)

    hdu = fits.PrimaryHDU(header=header, data=f[0].data[tuple(slice)])
    return hdu


def in_hull(points, x):
    n_points = len(points)
    n_dim = len(x)
    c = np.zeros(n_points)
    A = np.r_[points.T, np.ones((1, n_points))]
    b = np.r_[x, np.ones(1)]
    lp = linprog(c, A_eq=A, b_eq=b)
    return lp.success

class training_data_gen(object):
    def __init__(self, K, crop_size):
        self.K = K
        self.crop_size = crop_size

    def __call__(self, label_files, ref_images, datapacks):
        for label_file, ref_image, datapack in zip(label_files, ref_images, datapacks):
            label_file, ref_image, datapack = label_file.decode(), ref_image.decode(), datapack.decode()
            print("Getting data for", label_file, ref_image, datapack, self.K)
            with fits.open(ref_image, mode='readonly') as f:
                hdu = flatten(f)
                # data = hdu.data
                wcs = WCS(hdu.header)

            tec = np.load(datapack)['tec'].copy()
            tec_uncert = np.load(datapack)['tec_uncert'].copy()
            directions = np.load(datapack)['directions'].copy()
            # dp = DataPack(datapack, readonly=True)
            # dp.current_solset = 'directionally_referenced'
            # dp.select(pol=slice(0, 1, 1))
            # tec, axes = dp.tec
            _, Nd, Na, Nt = tec.shape
            # tec_uncert, _ = dp.weights_tec
            tec_uncert = np.maximum(0.1, np.where(np.isinf(tec_uncert), 1., tec_uncert))
            # _, directions = dp.get_directions(axes['dir'])
            # directions = np.stack([directions.ra.deg, directions.dec.deg], axis=1)
            directions = wcs.wcs_world2pix(directions, 0)

            __, nn_idx = cKDTree(directions).query(directions, k=self.K + 1)


            # Nd, Na, Nt
            human_flags = np.load(label_file)
            # Nd*Na,Nt, 1
            labels = human_flags.reshape((Nd * Na, Nt, 1)).astype(np.int32)
            mask = np.reshape(human_flags != -1, (Nd * Na, Nt, 1)).astype(np.int32)
            labels = np.where(labels == -1., 0., labels)

            # tec = np.pad(tec,[(0,0),(0,0), (0,0), (window_size, window_size)],mode='reflect')
            # tec_uncert = np.pad(tec_uncert,[(0,0),(0,0), (0,0), (window_size, window_size)],mode='reflect')

            inputs = []
            for d in range(Nd):
                # K+1, Na, Nt, 2
                input = np.stack([tec[0, nn_idx[d, :], :, :] / 10., np.log(tec_uncert[0, nn_idx[d, :], :, :])], axis=-1)
                # Na, Nt, (K+1)*2
                input = np.transpose(input, (1, 2, 0, 3)).reshape((Na, Nt, (self.K + 1) * 2))
                inputs.append(input)

            # Nd*Na,Nt, (K+1)*2
            inputs = np.concatenate(inputs, axis=0)
            for b in range(inputs.shape[0]):
                if np.sum(mask[b,:,:]) == 0:
                    # print("Skipping", b)
                    continue
                # print("Reading", b)
                while True:
                    start = np.random.choice(Nt - self.crop_size)
                    stop = start + self.crop_size
                    if np.sum(mask[b,start:stop,0]) == 0:
                        continue
                    break
                if np.random.uniform() < 0.5:
                    yield (inputs[b,start:stop:1,:], labels[b, start:stop:1, :], mask[b, start:stop:1, :])
                else:
                    yield (inputs[b,stop:start:-1,:], labels[b, stop:start:-1, :], mask[b, stop:start:-1, :])
        return

class eval_data_gen(object):
    def __init__(self, K):
        self.K = K

    def __call__(self, ref_images, datapacks):
        for ref_image, datapack in zip(ref_images, datapacks):
            ref_image, datapack = ref_image.decode(), datapack.decode()
            print("Getting data for", ref_image, datapack, self.K)
            with fits.open(ref_image, mode='readonly') as f:
                hdu = flatten(f)
                # data = hdu.data
                wcs = WCS(hdu.header)

            tec = np.load(datapack)['tec'].copy()
            tec_uncert = np.load(datapack)['tec_uncert'].copy()
            directions = np.load(datapack)['directions'].copy()
            # dp = DataPack(datapack, readonly=True)
            # dp.current_solset = 'directionally_referenced'
            # dp.select(pol=slice(0, 1, 1))
            # tec, axes = dp.tec
            _, Nd, Na, Nt = tec.shape
            # tec_uncert, _ = dp.weights_tec
            tec_uncert = np.maximum(0.1, np.where(np.isinf(tec_uncert), 1., tec_uncert))
            # _, directions = dp.get_directions(axes['dir'])
            # directions = np.stack([directions.ra.deg, directions.dec.deg], axis=1)
            directions = wcs.wcs_world2pix(directions, 0)

            __, nn_idx = cKDTree(directions).query(directions, k=self.K + 1)

            inputs = []
            for d in range(Nd):
                # K+1, Na, Nt, 2
                input = np.stack([tec[0, nn_idx[d, :], :, :] / 10., np.log(tec_uncert[0, nn_idx[d, :], :, :])], axis=-1)
                # Na, Nt, (K+1)*2
                input = np.transpose(input, (1, 2, 0, 3)).reshape((Na, Nt, (self.K + 1) * 2))
                inputs.append(input)

            # Nd*Na,Nt, (K+1)*2
            inputs = np.concatenate(inputs, axis=0)
            for b in range(inputs.shape[0]):
                yield (inputs[b,:,:],)
        return

# def build_training_dataset(label_file, ref_image, datapack, K=3):
#     """
#
#     :param label_file:
#     :param datapack:
#     :return:
#     """
#
#     print("Getting data for", label_file, ref_image, datapack, K)
#     with fits.open(ref_image, mode='readonly') as f:
#         hdu = flatten(f)
#         # data = hdu.data
#         wcs = WCS(hdu.header)
#
#     dp = DataPack(datapack, readonly=True)
#
#     dp.current_solset = 'directionally_referenced'
#     dp.select(pol=slice(0, 1, 1))
#     tec, axes = dp.tec
#     _, Nd, Na, Nt = tec.shape
#     tec_uncert, _ = dp.weights_tec
#     tec_uncert = np.where(np.isinf(tec_uncert), np.nanmean(tec_uncert), tec_uncert)
#     _, directions = dp.get_directions(axes['dir'])
#     directions = np.stack([directions.ra.deg, directions.dec.deg], axis=1)
#     directions = wcs.wcs_world2pix(directions, 0)
#
#     __, nn_idx = cKDTree(directions).query(directions, k=K + 1)
#
#     # Nd, Na, Nt
#     human_flags = np.load(label_file)
#     # Nd*Na,Nt, 1
#     labels = human_flags.reshape((Nd * Na, Nt, 1)).astype(np.int32)
#     mask = np.reshape(human_flags != -1, (Nd * Na, Nt, 1)).astype(np.int32)
#     labels = np.where(labels == -1., 0., labels)
#
#     assert np.all(labels >= 0)
#
#     # tec = np.pad(tec,[(0,0),(0,0), (0,0), (window_size, window_size)],mode='reflect')
#     # tec_uncert = np.pad(tec_uncert,[(0,0),(0,0), (0,0), (window_size, window_size)],mode='reflect')
#
#     inputs = []
#     for d in range(Nd):
#         # K+1, Na, Nt, 2
#         input = np.stack([tec[0, nn_idx[d, :], :, :] / 10., np.log(tec_uncert[0, nn_idx[d, :], :, :])], axis=-1)
#         # Na, Nt, (K+1)*2
#         input = np.transpose(input, (1, 2, 0, 3)).reshape((Na, Nt, (K + 1) * 2))
#         inputs.append(input)
#
#     # Nd*Na,Nt, (K+1)*2
#     inputs = np.concatenate(inputs, axis=0)
#     return [inputs, labels, mask]
#
#
#
# def build_eval_dataset(ref_image, datapack, K=3):
#     """
#
#     :param label_file:
#     :param datapack:
#     :return:
#     """
#
#     print("Getting data for", ref_image, datapack, K)
#     with fits.open(ref_image, mode='readonly') as f:
#         hdu = flatten(f)
#         # data = hdu.data
#         wcs = WCS(hdu.header)
#
#     dp = DataPack(datapack, readonly=True)
#
#     dp.current_solset = 'directionally_referenced'
#     dp.select(pol=slice(0, 1, 1))
#     tec, axes = dp.tec
#     _, Nd, Na, Nt = tec.shape
#     tec_uncert, _ = dp.weights_tec
#     tec_uncert = np.where(np.isinf(tec_uncert), np.nanmean(tec_uncert), tec_uncert)
#     tec_uncert = np.maximum(tec_uncert, 0.1)
#     _, directions = dp.get_directions(axes['dir'])
#     directions = np.stack([directions.ra.deg, directions.dec.deg], axis=1)
#     directions = wcs.wcs_world2pix(directions, 0)
#
#     __, nn_idx = cKDTree(directions).query(directions, k=K + 1)
#
#     inputs = []
#     for d in range(Nd):
#         # K+1, Na, Nt, 2
#         input = np.stack([tec[0, nn_idx[d, :], :, :] / 10., np.log(tec_uncert[0, nn_idx[d, :], :, :])], axis=-1)
#         # Na, Nt, (K+1)*2
#         input = np.transpose(input, (1, 2, 0, 3)).reshape((Na, Nt, (K + 1) * 2))
#         inputs.append(input)
#
#     # Nd*Na,Nt, (K+1)*2
#     inputs = np.concatenate(inputs, axis=0)
#
#     return [inputs]


def get_output_bias(label_files):
    num_pos = 0
    num_neg = 0
    for label_file in label_files:
        human_flags = np.load(label_file)
        num_pos += np.sum(human_flags == 1)
        num_neg += np.sum(human_flags == 0)
    pos_weight = num_neg / num_pos
    bias = np.log(num_pos) - np.log(num_neg)
    return bias, pos_weight


class Classifier(object):
    def __init__(self, L=4, K=3, n_features=16, batch_size=16, graph=None, output_bias=0., pos_weight=1., crop_size=60):
        if graph is None:
            graph = tf.Graph()
        self.graph = graph
        self.K = K
        N = (K + 1) * 2
        self.L = L
        self.crop_size = crop_size
        self.n_features = n_features
        with self.graph.as_default():
            self.label_files_pl = tf.placeholder(tf.string, shape=[None], name='label_files')
            self.datapacks_pl = tf.placeholder(tf.string, shape=[None], name='datapacks')
            self.ref_images_pl = tf.placeholder(tf.string, shape=[None], name='ref_images')

            ###
            # train/test inputs

            dataset = tf.data.Dataset.from_tensors((self.label_files_pl, self.ref_images_pl, self.datapacks_pl))
            dataset = dataset.interleave(lambda  label_files, ref_images, datapacks:
                                         tf.data.Dataset.from_generator(
                                             training_data_gen(self.K, self.crop_size),
                                             output_types=(tf.float32, tf.int32, tf.int32),
                                             output_shapes=((self.crop_size, N),
                                                            (self.crop_size, 1),
                                                            (self.crop_size, 1),),
                                             args=(label_files, ref_images, datapacks)),
                                         cycle_length=1,
                                         block_length=2
                                         )
            train_dataset = dataset.shard(2, 0).shuffle(1000)
            train_dataset = train_dataset.batch(batch_size=batch_size, drop_remainder=True)

            test_dataset = dataset.shard(2, 1).shuffle(1000)
            test_dataset = test_dataset.batch(batch_size=batch_size, drop_remainder=True)

            iterator_tensor = train_dataset.make_initializable_iterator()
            self.train_init = iterator_tensor.initializer
            self.train_inputs, self.train_labels, self.train_mask = iterator_tensor.get_next()
            # self.train_inputs.set_shape([None, None, N])

            iterator_tensor = test_dataset.make_initializable_iterator()
            self.test_init = iterator_tensor.initializer
            self.test_inputs, self.test_labels, self.test_mask = iterator_tensor.get_next()
            # self.test_inputs.set_shape([None, None, N])

            ###
            # eval inputs
            dataset = tf.data.Dataset.from_tensors((self.ref_images_pl, self.datapacks_pl))
            dataset = dataset.interleave(lambda ref_images, datapacks:
                                         tf.data.Dataset.from_generator(
                                             eval_data_gen(self.K),
                                             output_types=(tf.float32,),
                                             output_shapes=((None, N),),
                                             args=(ref_images, datapacks)),
                                         cycle_length=1,
                                         block_length=1
                                         )
            eval_dataset = dataset.batch(batch_size=batch_size, drop_remainder=False)

            iterator_tensor = eval_dataset.make_initializable_iterator()
            self.eval_init = iterator_tensor.initializer
            self.eval_inputs, = iterator_tensor.get_next()
            # self.eval_inputs.set_shape([None, None, N])

            ###
            # outputs

            train_outputs = self.build_model(self.train_inputs, output_bias=output_bias)
            test_outputs = self.build_model(self.test_inputs, output_bias=output_bias)
            eval_outputs = self.build_model(self.eval_inputs, output_bias=output_bias)

            num_models = len(tf.unstack(train_outputs))

            self.thresholds = tf.Variable(0.5*np.ones(num_models), shape=[num_models], dtype=tf.float32)
            self.thresholds_pl = tf.placeholder(tf.float32, [num_models])
            self.assign_thresholds = tf.assign(self.thresholds, self.thresholds_pl)

            labels_ext = tf.broadcast_to(self.train_labels, tf.shape(train_outputs))
            # mask_ext = tf.broadcast_to(self.train_mask, tf.shape(train_outputs))
            self.train_pred_probs = tf.nn.sigmoid(train_outputs)
            # self.train_conf_mat = tf.math.confusion_matrix(tf.reshape(labels_ext, (-1,)),
            #                                                tf.reshape(self.train_pred_probs > self.thresholds[:, None, None, None], (-1,)),
            #                                                weights=tf.reshape(mask_ext, (-1,)),
            #                                                num_classes=2, dtype=tf.float32)

            self.train_conf_mat = tf.math.confusion_matrix(tf.reshape(self.train_labels, (-1,)),
                                                           tf.reshape(
                                                               tf.reduce_mean(tf.cast(self.train_pred_probs > self.thresholds[:, None, None,
                                                                                       None], tf.float32), 0) >= 0.5, (-1,)),
                                                           weights=tf.reshape(self.train_mask, (-1,)),
                                                           num_classes=2, dtype=tf.float32)
            loss = tf.nn.weighted_cross_entropy_with_logits(labels=tf.cast(labels_ext, train_outputs.dtype), logits=train_outputs,
                                                            pos_weight=pos_weight)
            self.train_loss = tf.reduce_mean(loss * tf.cast(self.train_mask, loss.dtype))

            labels_ext = tf.broadcast_to(self.test_labels, tf.shape(test_outputs))
            # mask_ext = tf.broadcast_to(self.test_mask, tf.shape(test_outputs))
            self.test_pred_probs = tf.nn.sigmoid(test_outputs)
            self.test_conf_mat = tf.math.confusion_matrix(tf.reshape(self.test_labels, (-1,)),
                                                          tf.reshape(tf.reduce_mean(tf.cast(self.test_pred_probs > self.thresholds[:, None, None, None], tf.float32), 0)>=0.5, (-1,)),
                                                          weights=tf.reshape(self.train_mask, (-1,)),
                                                          num_classes=2, dtype=tf.float32)
            loss = tf.nn.weighted_cross_entropy_with_logits(labels=tf.cast(labels_ext, test_outputs.dtype), logits=test_outputs,
                                                            pos_weight=pos_weight)
            self.test_loss = tf.reduce_mean(loss * tf.cast(self.test_mask, loss.dtype))

            self.eval_pred_probs = tf.nn.sigmoid(eval_outputs)
            self.eval_pred_labels = tf.reduce_mean(tf.cast(self.eval_pred_probs > self.thresholds[:, None, None, None],
                                                          tf.float32), 0) >= 0.5

            self.global_step = tf.Variable(0, trainable=False)
            self.opt = tf.train.AdamOptimizer().minimize(self.train_loss, global_step=self.global_step)

    def conf_mat_to_str(self, conf_mat):
        tn = conf_mat[0, 0]
        fp = conf_mat[0, 1]
        fn = conf_mat[1, 0]
        tp = conf_mat[1, 1]
        T = tp + fn
        F = tn + fp
        acc = (tp + tn) / (T + F)
        rel_acc = (2. * acc - 1.) * 100.
        fpr = fp / F
        rel_fpr = (fpr / (0.5 * F / T) - 1.) * 100.
        fnr = fn / T
        rel_fnr = (fnr / (0.5 * T / F) - 1.) * 100.
        with np.printoptions(precision=2):
            return "[{} {} | {} {}] ACC: {} [{}% baseline] FPR: {} [{}% baseline] FNR: {} [{}% baseline]".format(tp, fn, fp, tn, acc, rel_acc, fpr,
                                                                                                 rel_fpr, fnr, rel_fnr)

    def train_model(self, label_files, ref_images, datapacks, epochs=10, print_freq=100, working_dir='./training'):
        os.makedirs(working_dir, exist_ok=True)
        with tf.Session(graph=self.graph) as sess:
            saver = tf.train.Saver()
            print("initialising valiables")
            sess.run(tf.initialize_variables(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)))
            print("restoring if possibe")
            try:
                saver.restore(sess, tf.train.latest_checkpoint(working_dir))
            except:
                pass
            print("Running {} epochs".format(epochs))

            for epoch in range(epochs):
                print("epoch", epoch)
                print("init data for training")
                sess.run([self.train_init, self.test_init],
                         {self.label_files_pl: label_files,
                          self.ref_images_pl: ref_images,
                          self.datapacks_pl: datapacks})
                epoch_train_conf_mat = np.zeros((2, 2))
                epoch_test_conf_mat = np.zeros((2, 2))
                epoch_train_loss = []
                epoch_test_loss = []
                print("Run loop")
                epoch_test_preds = []
                epoch_test_labels = []
                epoch_test_mask = []
                batch = 0
                while True:
                    try:
                        _, train_loss, train_conf_mat, test_loss, test_conf_mat, global_step, test_preds, test_labels, \
                        test_mask = sess.run(
                            [self.opt, self.train_loss, self.train_conf_mat, self.test_loss, self.test_conf_mat,
                             self.global_step, self.test_pred_probs, self.test_labels, self.test_mask])
                        epoch_train_conf_mat += train_conf_mat
                        epoch_test_conf_mat += test_conf_mat
                        epoch_train_loss.append(train_loss)
                        epoch_test_loss.append(test_loss)
                        epoch_test_preds.append(test_preds)
                        epoch_test_labels.append(test_labels)
                        epoch_test_mask.append(test_mask)
                        if global_step % print_freq == 0:
                            with np.printoptions(precision=2):
                                print("Step {:04d} Epoch {} batch {} train loss {} test loss {}".format(global_step,
                                                                                                        epoch, batch,
                                                                                                        train_loss,
                                                                                                        test_loss))
                                print("\tTrain {}".format(self.conf_mat_to_str(train_conf_mat)))
                                print("\tTest  {}".format(self.conf_mat_to_str(test_conf_mat)))
                        batch += 1
                    except tf.errors.OutOfRangeError:
                        break
                epoch_test_preds = np.concatenate(epoch_test_preds, axis=1)
                epoch_test_labels = np.concatenate(epoch_test_labels, axis=0)
                epoch_test_labels = np.where(epoch_test_labels == -1, 0, epoch_test_labels)
                epoch_test_mask = np.concatenate(epoch_test_mask, axis=0)

                print("Epoch {} train loss: {} +- {} test loss: {} +- {}".format(epoch, np.mean(epoch_train_loss),
                                                                                 np.std(epoch_train_loss),
                                                                                 np.mean(epoch_test_loss),
                                                                                 np.std(epoch_test_loss)))
                print("Epoch {} train {} ".format(epoch, self.conf_mat_to_str(epoch_train_conf_mat)))
                print("Epoch {} test  {} ".format(epoch, self.conf_mat_to_str(epoch_test_conf_mat)))
                opt_thresholds = []
                for m in range(epoch_test_preds.shape[0]):
                    fpr, tpr, thresholds = roc_curve(epoch_test_labels.flatten(), epoch_test_preds[m,...].flatten(),
                                                     sample_weight=epoch_test_mask.flatten())
                    which =  np.argmax(tpr - fpr)
                    opt_thresholds.append(thresholds[which])
                    # plt.scatter(fpr, tpr)
                    # plt.scatter(fpr[which], tpr[which], c='red')
                    # plt.title("Model {]: {}".format(m, thresholds[which]))
                    # plt.show()
                    # plt.close('all')
                print("New opt thresholds: {}".format(opt_thresholds))
                sess.run(self.assign_thresholds,{self.thresholds_pl:opt_thresholds})
                print('Saving...')
                save_path = saver.save(sess, self.save_path(working_dir), global_step=self.global_step)
                print("Saved to {}".format(save_path))

    def save_path(self, working_dir):
        return os.path.join(working_dir, 'model-K{:02d}.ckpt'.format(self.K))



    def eval_model(self, ref_images, datapacks, working_dir='./training'):
        with tf.Session(graph=self.graph) as sess:
            saver = tf.train.Saver()
            saver.restore(sess,tf.train.latest_checkpoint(working_dir))
            all_predictions = []
            for ref_image, datapack in zip(ref_images, datapacks):
                sess.run(self.eval_init,
                         {self.ref_images_pl: [ref_image],
                          self.datapacks_pl: [datapack]})
                predictions = []
                while True:
                    try:
                        winner = sess.run(self.eval_pred_labels)
                        predictions.append(winner)
                    except tf.errors.OutOfRangeError:
                        break
                predictions = np.concatenate(predictions, axis=0)
                print("{} Predictions: {}".format(datapack, predictions.shape))
                all_predictions.append(predictions)
            return all_predictions

    def build_model(self, inputs, output_bias=0.):
        with tf.variable_scope('classifier', reuse=tf.AUTO_REUSE):
            num = 0
            features = tf.layers.conv1d(inputs, self.n_features, [1], strides=1, padding='same', activation=None,
                                        name='conv_{:02d}'.format(num))
            num += 1
            outputs = []
            for s in [1]:
                for d in [1, 2, 3, 4]:
                    if s > 1 and d > 1:
                        continue
                    for pool in [tf.layers.average_pooling1d, tf.layers.max_pooling1d]:
                        h = [features]
                        for i in range(self.L):
                            u = tf.layers.conv1d(h[i], self.n_features, [3], s, padding='same', dilation_rate=d,
                                                 activation=tf.nn.relu,
                                                 use_bias=True, name='conv_{:02d}'.format(num))
                            num += 1
                            u = pool(u, pool_size=3, strides=1, padding='same')
                            h.append(u - h[i])
                        outputs.append(h[-1])
            # S, Nd*Na, Nt, 1
            outputs = tf.stack(
                [tf.layers.conv1d(o, 1, [1], padding='same', name='conv_{:02d}'.format(num), use_bias=False) for o in
                 outputs], axis=0)
            output_bias = tf.Variable(output_bias, dtype=tf.float32, trainable=False)
            # outputs -= tf.reduce_mean(outputs, axis=-1,keepdims=True)
            outputs += output_bias
            num += 1
            return outputs

    def _augment(self, inputs, labels, mask):
        sizes = [(1 + self.K) * 2, 1, 1]
        c = np.cumsum(sizes)
        N = sum(sizes)
        #B, Nt, N
        large = tf.concat([inputs, labels, mask], axis=-1)
        large = tf.image.random_flip_left_right(
            tf.image.random_crop(large, (tf.shape(inputs)[0], self.crop_size, N)))
        inputs, labels, mask = large[..., :c[0]], large[..., c[0]:c[1]], large[..., c[1]:c[2]]
        return [inputs, labels, mask]


def click_through(save_file, datapack, ref_image, model_dir, classifier, reset=False):
    with fits.open(ref_image) as f:
        hdu = flatten(f)
        wcs = WCS(hdu.header)
    window = 20

    dp = DataPack(datapack, readonly=True)

    dp.current_solset = 'directionally_referenced'
    dp.select(pol=slice(0, 1, 1))
    tec, axes = dp.tec
    _, Nd, Na, Nt = tec.shape
    tec_uncert, _ = dp.weights_tec
    _, directions = dp.get_directions(axes['dir'])
    directions = np.stack([directions.ra.deg, directions.dec.deg], axis=1)
    directions = wcs.wcs_world2pix(directions, 0)
    _, times = dp.get_times(axes['time'])
    times = times.mjd * 86400.
    times -= times[0]
    times /= 3600.

    window_time = times[window]

    xmin = directions[:, 0].min()
    xmax = directions[:, 0].max()
    ymin = directions[:, 1].min()
    ymax = directions[:, 1].max()

    radius = xmax - xmin

    ref_dir = directions[0:1, :]

    _, guess_flags = filter_tec_dir(tec[0, ...], directions, init_y_uncert=None, min_res=8.)
    if os.path.isfile(save_file) and not reset:
        human_flags = np.load(save_file)
    else:
        human_flags = -1 * np.ones([Nd, Na, Nt], np.int32)
    guess_flags = np.where(human_flags != -1, human_flags, guess_flags)

    # compute Voronoi tesselation
    vor = Voronoi(directions)

    __, nn_idx = cKDTree(directions).query(directions, k=4)

    regions, vertices = voronoi_finite_polygons_2d(vor, radius)

    fig = plt.figure(constrained_layout=False, figsize=(12, 12))

    gs = fig.add_gridspec(3, 2)
    time_ax = fig.add_subplot(gs[0, :])
    time_ax.set_xlabel('time [hours]')
    time_ax.set_ylabel('DDTEC [mTECU]')
    vline = time_ax.plot([0, 0], [-1, 1], c='red', alpha=0.5)[0]

    time_plots = [time_ax.plot(np.arange(window * 2), 0. * np.arange(window * 2), c='black')[0] for _ in range(4)]

    dir_ax = fig.add_subplot(gs[1:, :], projection=wcs)
    dir_ax.coords[0].set_axislabel('Right Ascension (J2000)')
    dir_ax.coords[1].set_axislabel('Declination (J2000)')
    dir_ax.coords.grid(True, color='grey', ls='solid')
    polygons = []
    cmap = plt.cm.get_cmap('PuOr')
    norm = plt.Normalize(-1., 1.)
    colors = np.zeros(Nd)
    # colorize
    for color, region in zip(colors, regions):
        if np.size(color) == 1:
            if norm is None:
                color = cmap(color)
            else:
                color = cmap(norm(color))
        polygon = vertices[region]
        polygons.append(dir_ax.fill(*zip(*polygon), color=color, alpha=1., linewidth=4, edgecolor='black')[0])

    dir_ax.scatter(ref_dir[:, 0], ref_dir[:, 1], marker='*', color='black', zorder=19)

    # plt.plot(points[:,0], points[:,1], 'ko')
    dir_ax.set_xlim(vor.min_bound[0] - 0.1 * radius, vor.max_bound[0] + 0.1 * radius)
    dir_ax.set_ylim(vor.min_bound[1] - 0.1 * radius, vor.max_bound[1] + 0.1 * radius)

    def onkeyrelease(event):
        _, a, t, norm, _, _ = loc
        print('Pressed {} ({}, {})'.format(event.key, event.xdata, event.ydata))
        if event.key == 'n':
            print("Saving... going to next.")
            human_flags[:, a, t] = np.where(human_flags[:, a, t] == -1, 0, human_flags[:, a, t])
            np.save(save_file, human_flags)
            next_loc = min(loc[0] + 1, len(order))
            load_data(next_loc)
        if event.key == 'b':
            print("Saving... going to back.")
            human_flags[:, a, t] = np.where(human_flags[:, a, t] == -1, 0, human_flags[:, a, t])
            np.save(save_file, human_flags)
            next_loc = max(loc[0] - 1, 0)
            load_data(next_loc)
        if event.key == 'x':
            print("Exit")
            # np.save(save_file, human_flags)
            plt.close('all')
        if event.key == 'l':
            print("Learning one epoch")
            classifier.train_model([save_file], [ref_image], [datapack.replace('.h5', '.npz')], epochs=1, print_freq=100,
                          working_dir=model_dir)
        if event.key == 'p':
            print("Predicting with neural net...")
            # c = Classifier(L=5, K=6, n_features=24, crop_size=250, batch_size=16, output_bias=output_bias,
            #                pos_weight=pos_weight)
            # c.train_model(label_files, linked_ref_images, linked_datapack_npzs, epochs=100, print_freq=100,
            #               working_dir=os.path.join(working_dir, 'model'))
            pred = classifier.eval_model([ref_image], [datapack.replace('.h5', '.npz')],
                                         working_dir=model_dir)[0]
            guess_flags[...] = pred.reshape((Nd, Na, Nt))
            search, order = rebuild_order()
            loc[0] = 0
            loc[4] = search
            loc[5] = order
            load_data(loc[0])

        if event.key == 'c':
            print("Copying over predicted...")
            human_flags[:, a, t] = np.where(human_flags[:, a, t] == -1, guess_flags[:, a, t], human_flags[:, a, t])
            load_data(loc[0])


    def onclick(event):
        _, a, t, norm, _, _ = loc
        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              ('double' if event.dblclick else 'single', event.button,
               event.x, event.y, event.xdata, event.ydata))
        for i, region in enumerate(regions):
            point = i
            if in_hull(vertices[region], np.array([event.xdata, event.ydata])):
                print("In region {} (point {})".format(i, point))
                if event.button == 1:
                    print("Changing {}".format(human_flags[point, a, t]))
                    if human_flags[point, a, t] == -1 or human_flags[point, a, t] == 0:
                        human_flags[point, a, t] = 1
                        polygons[i].set_edgecolor('red')
                    elif human_flags[point, a, t] == 1:
                        human_flags[point, a, t] = 0
                        polygons[i].set_edgecolor('green')
                    polygons[i].set_zorder(11)
                    print("to {}".format(human_flags[point, a, t]))
                if event.button == 3 or event.button == 1:
                    start = max(0, t - window)
                    stop = min(Nt, t + window)
                    for n in range(4):
                        closest_idx = nn_idx[point, n]
                        time_plot = time_plots[n]
                        time_plot.set_data(times[start:stop], tec[0, closest_idx, a, start:stop])
                        if n == 0:
                            time_plot.set_color('black')
                        else:
                            time_plot.set_color(cmap(norm(tec[0, closest_idx, a, t])))
                    time_ax.set_xlim(times[t] - window_time, times[t] + window_time)
                    time_ax.set_ylim(tec[0, point, a, start:stop].min() - 5., tec[0, point, a, start:stop].max() + 5.)
                    vline.set_data([times[t], times[t]],
                                   [tec[0, point, a, start:stop].min() - 5., tec[0, point, a, start:stop].max() + 5.])
                fig.canvas.draw()
                # break

    cid_click = fig.canvas.mpl_connect('button_press_event', onclick)
    cid_key = fig.canvas.mpl_connect('key_release_event', onkeyrelease)

    # Na, Nt

    def rebuild_order():
        mask = np.logical_and(np.any(guess_flags, axis=0), np.logical_not(np.any(human_flags>=0)))
        search_first = np.where(mask)
        search_second = np.where(np.logical_not(mask))
        search = [list(sf) + list(ss) for sf, ss in zip(search_first, search_second)]
        order = list(np.random.choice(len(search_first[0]), len(search_first[0]), replace=False)) + \
                list(len(search_first[0]) + np.random.choice(len(search_second[0]), len(search_second[0]), replace=False))
        return search, order

    search, order = rebuild_order()
    loc = [0, 0, 0, plt.Normalize(-1., 1.), search, order]

    def load_data(next_loc):
        loc[0] = next_loc
        search = loc[4]
        order = loc[5]
        o = order[next_loc]
        a = search[0][o]
        t = search[1][o]
        loc[1] = a
        loc[2] = t

        print("Looking at ant{:02d} and time {}".format(a, t))
        vmin, vmax = np.min(tec[0, :, a, t]), np.max(tec[0, :, a, t])
        vmin, vmax = min(vmin, -vmax), max(vmax, -vmin)
        norm = plt.Normalize(vmin, vmax)
        loc[3] = norm
        for i, p in enumerate(polygons):
            p.set_facecolor(cmap(norm(tec[0, i, a, t])))
            if human_flags[i, a, t] == 0:
                p.set_edgecolor('green')
                p.set_zorder(10)
            elif human_flags[i, a, t] == 1:
                p.set_edgecolor('red')
                p.set_zorder(11)
            elif guess_flags[i, a, t]:
                p.set_edgecolor('cyan')
                p.set_zorder(11)
            else:
                p.set_edgecolor('black')
                p.set_zorder(10)
        # human_flags[:, a, t] = 0
        fig.canvas.draw()

    load_data(0)
    plt.show()
    return False


if __name__ == '__main__':
    # dp = '/net/nederrijn/data1/albert/screens/root/L562061/download_archive/L562061_DDS4_full_merged.h5'
    # ref_img = '/net/nederrijn/data1/albert/screens/root/L562061/download_archive/image_full_ampphase_di_m.NS.mask01.fits'
    # click_through(dp, ref_img)

    import os, glob

    working_dir = os.path.join('/home/albert/git/bayes_gain_screens/debug', 'outlier_detection')
    os.makedirs(working_dir, exist_ok=True)
    datapacks = glob.glob('/home/albert/store/root_dense/L*/download_archive/L*_DDS4_full_merged.h5')
    # ref_images = [os.path.join(os.path.dirname(f), 'image_full_ampphase_di_m.NS.app.restored.fits') for f in datapacks]
    ref_images = ['/home/albert/store/lockman/archive/image_full_ampphase_di_m.NS.app.restored.fits'] * len(datapacks)
    label_files = []
    linked_datapacks = []
    linked_ref_images = []
    linked_datapack_npzs = []

    classifier = Classifier(L=5, K=6, n_features=24, crop_size=250, batch_size=16, output_bias=0.,
                           pos_weight=0.)

    for dp, ref_img in zip(datapacks, ref_images):
        linked_datapack = os.path.join(working_dir, os.path.basename(os.path.abspath(dp)))
        if os.path.islink(linked_datapack):
            os.unlink(linked_datapack)
        print("Linking {} -> {}".format(os.path.abspath(dp), linked_datapack))
        os.symlink(os.path.abspath(dp), linked_datapack)

        linked_ref_image = linked_datapack.replace('.h5', '.ref_image.fits')
        if os.path.islink(linked_ref_image):
            os.unlink(linked_ref_image)
        print("Linking {} -> {}".format(os.path.abspath(ref_img), linked_ref_image))
        os.symlink(os.path.abspath(ref_img), linked_ref_image)

        save_file = linked_datapack.replace('.h5', '.labels.npy')
        label_files.append(save_file)
        linked_datapacks.append(linked_datapack)
        linked_ref_images.append(linked_ref_image)

        if click_through(save_file, linked_datapack, linked_ref_image,
                         model_dir=os.path.join(working_dir, 'model'), classifier=classifier, reset=False):
            break

        linked_datapack_npz = linked_datapack.replace('.h5', '.npz')
        if not os.path.isfile(linked_datapack_npz):
            dp = DataPack(dp, readonly=True)
            dp.current_solset='directionally_referenced'
            dp.select(pol=slice(0,1,1))
            tec, axes = dp.tec
            tec_uncert, _ = dp.weights_tec
            _, directions = dp.get_directions(axes['dir'])

            np.savez(linked_datapack_npz, tec=tec, tec_uncert=tec_uncert,
                     directions=np.stack([directions.ra.deg, directions.dec.deg], axis=1))
        linked_datapack_npzs.append(linked_datapack_npz)


    # output_bias, pos_weight = get_output_bias(label_files)
    # print("Output bias: {}".format(output_bias))
    # print("Pos weight: {}".format(pos_weight))
    # c = Classifier(L=5, K=6, n_features=24, crop_size=250, batch_size=16, output_bias=output_bias, pos_weight=pos_weight)
    # c.train_model(label_files, linked_ref_images, linked_datapack_npzs, epochs=100, print_freq=100,
    #               working_dir=os.path.join(working_dir, 'model'))
    # c.eval_model(linked_ref_images, linked_datapack_npzs,working_dir=os.path.join(working_dir, 'model'))

