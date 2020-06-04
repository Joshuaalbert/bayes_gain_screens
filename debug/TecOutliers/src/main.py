import tensorflow.compat.v1 as tf
import numpy as np
from sklearn.metrics import roc_curve
import pylab as plt
import os, sys, glob, argparse
from timeit import default_timer
from graph_nets.modules import _unsorted_segment_softmax
from graph_nets.graphs import GraphsTuple
import sonnet as snt
from graph_nets import modules, blocks
from graph_nets import utils_tf
import logging

tf.get_logger().setLevel('ERROR')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def batched_tensor_to_fully_connected_graph_tuple_dynamic(nodes_tensor, pos=None, globals=None):
    """
    Convert tensor with batch dim to batch of GraphTuples.
    :param nodes_tensor: [B, num_nodes, F] Tensor to turn into nodes. F must be statically known.
    :param pos: [B, num_nodes, D] Tensor to calculate edge distance. D must be statically known.
    :param globals: [B, G] Tensor to use as global. G must be statically known.
    :return: GraphTuple with batch of fully connected graphs
    """
    shape = tf.shape(nodes_tensor)
    batch_size, num_nodes = shape[0], shape[1]
    F = nodes_tensor.shape.as_list()[-1]
    graphs_with_nodes = GraphsTuple(n_node=tf.fill([batch_size], num_nodes),
                                    n_edge=tf.fill([batch_size], 0),
                                    nodes=tf.reshape(nodes_tensor, [batch_size * num_nodes, F]),
                                    edges=None, globals=None, receivers=None, senders=None)
    graphs_tuple_with_nodes_connectivity = utils_tf.fully_connect_graph_dynamic(
        graphs_with_nodes, exclude_self_edges=False)

    if pos is not None:
        D = pos.shape.as_list()[-1]
        graphs_with_position = graphs_tuple_with_nodes_connectivity.replace(
            nodes=tf.reshape(pos, [batch_size * num_nodes, D]))
        edge_distances = (
                blocks.broadcast_receiver_nodes_to_edges(graphs_with_position) -
                blocks.broadcast_sender_nodes_to_edges(graphs_with_position))
        graphs_with_nodes_edges = graphs_tuple_with_nodes_connectivity.replace(edges=edge_distances)
    else:
        graphs_with_nodes_edges = utils_tf.set_zero_edge_features(graphs_tuple_with_nodes_connectivity, 1,
                                                                  dtype=nodes_tensor.dtype)

    if globals is not None:
        graphs_with_nodes_edges_globals = graphs_with_nodes_edges.replace(globals=globals)
    else:
        graphs_with_nodes_edges_globals = utils_tf.set_zero_global_features(
            graphs_with_nodes_edges, global_size=1)

    return graphs_with_nodes_edges_globals


def make_mlp_model(layer_sizes, residual=False):
    """Instantiates a new MLP, followed by LayerNorm.
    The parameters of each new MLP are not shared with others generated by
    this function.
    Returns:
      A Sonnet module which contains the MLP and LayerNorm.
    """

    def func():
        layers = []
        for s in layer_sizes:
            layers.append(snt.Linear(s, allow_many_batch_dims=True))
            layers.append(tf.nn.relu)
        layers.append(snt.LayerNorm())
        if residual:
            return snt.Residual(snt.Sequential(layers))
        else:
            return snt.Sequential(layers)

    return func


class MLPGraphIndependent(snt.AbstractModule):
    """GraphIndependent with MLP edge, node, and global models."""

    def __init__(self, layer_sizes, name="MLPGraphIndependent"):
        super(MLPGraphIndependent, self).__init__(name=name)
        with self._enter_variable_scope():
            self._network = modules.GraphIndependent(
                edge_model_fn=make_mlp_model(layer_sizes),
                node_model_fn=make_mlp_model(layer_sizes),
                global_model_fn=make_mlp_model(layer_sizes))

    def _build(self, inputs):
        return self._network(inputs)


class MLPGraphNetwork(snt.AbstractModule):
    """GraphNetwork with MLP edge, node, and global models."""

    def __init__(self, layer_sizes, name="MLPGraphNetwork"):
        super(MLPGraphNetwork, self).__init__(name=name)
        with self._enter_variable_scope():
            self._network = modules.GraphNetwork(make_mlp_model(layer_sizes),
                                                 make_mlp_model(layer_sizes),
                                                 make_mlp_model(layer_sizes))

    def _build(self, inputs):
        return self._network(inputs)


class EncodeProcessDecode(snt.AbstractModule):
    """Full encode-process-decode model.
    The model we explore includes three components:
    - An "Encoder" graph net, which independently encodes the edge, node, and
      global attributes (does not compute relations etc.).
    - A "Core" graph net, which performs N rounds of processing (message-passing)
      steps. The input to the Core is the concatenation of the Encoder's output
      and the previous output of the Core (labeled "Hidden(t)" below, where "t" is
      the processing step).
    - A "Decoder" graph net, which independently decodes the edge, node, and
      global attributes (does not compute relations etc.), on each message-passing
      step.
                        Hidden(t)   Hidden(t+1)
                           |            ^
              *---------*  |  *------*  |  *---------*
              |         |  |  |      |  |  |         |
    Input --->| Encoder |  *->| Core |--*->| Decoder |---> Output(t)
              |         |---->|      |     |         |
              *---------*     *------*     *---------*
    """

    def __init__(self,
                 layer_sizes,
                 edge_output_size=None,
                 node_output_size=1,
                 global_output_size=None,
                 name="EncodeProcessDecode"):
        super(EncodeProcessDecode, self).__init__(name=name)
        self._encoder = MLPGraphIndependent(layer_sizes)
        self._core = MLPGraphNetwork(layer_sizes)
        self._decoder = MLPGraphIndependent(layer_sizes)
        # Transforms the outputs into the appropriate shapes.
        if edge_output_size is None:
            edge_fn = None
        else:
            edge_fn = lambda: snt.Linear(edge_output_size, allow_many_batch_dims=True, name="edge_output")
        if node_output_size is None:
            node_fn = None
        else:
            node_fn = lambda: snt.Linear(node_output_size, allow_many_batch_dims=True, name="node_output")
        if global_output_size is None:
            global_fn = None
        else:
            global_fn = lambda: snt.Linear(global_output_size, allow_many_batch_dims=True, name="global_output")
        with self._enter_variable_scope():
            self._output_transform = modules.GraphIndependent(edge_fn, node_fn,
                                                              global_fn)

    def _build(self, input_op, num_processing_steps):
        latent = self._encoder(input_op)
        latent0 = latent
        output_ops = []
        for _ in range(num_processing_steps):
            core_input = utils_tf.concat([latent0, latent], axis=1)
            _latent = self._core(core_input)#residual
            latent = _latent.replace(nodes=_latent.nodes + latent.nodes)
            decoded_op = self._decoder(latent)
            output_ops.append(self._output_transform(decoded_op))
        return output_ops

class Model(tf.keras.Model):
    def __init__(self, class_bias, rate=0.1):
        super(Model, self).__init__()
        self.class_bias = tf.constant(class_bias, dtype=tf.float32)

        self.project = tf.keras.layers.Conv2D(8, 1, activation=None)

        self.conv = [
            tf.keras.layers.Conv2D(8, (3, 1), padding='same', activation=None),
            tf.keras.layers.Conv2D(8, (3, 1), padding='same', activation=None),
            tf.keras.layers.Conv2D(8, (3, 1), padding='same', activation=None),
            tf.keras.layers.Conv2D(8, (3, 1), padding='same', activation=None)
        ]

        self.dropout = [tf.keras.layers.Dropout(rate),
                        tf.keras.layers.Dropout(rate),
                        tf.keras.layers.Dropout(rate),
                        tf.keras.layers.Dropout(rate)
                        ]

        self.layer_norm = [tf.keras.layers.BatchNormalization(renorm=True),
                           tf.keras.layers.BatchNormalization(renorm=True),
                           tf.keras.layers.BatchNormalization(renorm=True),
                           tf.keras.layers.BatchNormalization(renorm=True)
                           ]

        self.graph_network = EncodeProcessDecode([16, 16])

    @property
    def all_trainable_variables(self):
        return list(self.trainable_variables) + list(self.graph_network.trainable_variables)

    def call(self, tec, cal_pos, ant_pos, training=False):
        """

        :param tec: [B, Nt, Nd, 1]
        :param cal_pos: [B, Nt, Nd, D]
        :param ant_pos: [B, Nt, Na]
        :param training: bool
        :return: logits [B, Nt, Nd, 1]
        """
        ###
        # temporal CNN

        output = self.project(tec)
        saves = [output]
        for conv, dropout, norm in zip(self.conv, self.dropout, self.layer_norm):
            output = dropout(norm(tf.nn.relu(conv(output)), training=training), training=training) + output
            saves.append(output)
            print(output)
        # [B, Nt, Nd, 8]
        shape = tf.shape(output)
        B, Nt, Nd, _ = shape[0], shape[1], shape[2], shape[3]
        F = output.shape.as_list()[-1]

        # B*Nt, Nd, 8
        output = tf.reshape(output, (B * Nt, Nd, F))
        D = cal_pos.shape.as_list()[-1]
        cal_pos = tf.reshape(cal_pos, (B * Nt, Nd, D))
        G = ant_pos.shape.as_list()[-1]
        ant_pos = tf.reshape(ant_pos, (B * Nt, G))
        print(output, cal_pos, ant_pos)

        graph = batched_tensor_to_fully_connected_graph_tuple_dynamic(output, pos=cal_pos, globals=ant_pos)

        graph_roll = self.graph_network(graph, 5)
        logits = tf.reshape(graph_roll[-1].nodes, [B, Nt, Nd, 1]) + self.class_bias

        return logits


class TrainingDataGen(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, datapacks):
        for datapack in datapacks:
            datapack = datapack.decode()
            print("Getting data for", datapack)

            tec = np.load(datapack)['tec'].copy()
            _, Nd, Na, Nt = tec.shape
            directions = np.load(datapack)['directions'].copy()
            ref_directions = directions - directions[0:1, :]
            ref_dist = np.load(datapack)['ref_dist'].copy()  # Na,3
            # Nd, Na, Nt
            human_flags = np.load(datapack)['human_flags'].copy().transpose((1, 2, 0)).reshape((Na, Nt, Nd, 1))

            # Nd, Na, Nt -> Na, Nt, Nd
            mask = (human_flags != -1).astype(np.float32)
            labels = np.where(mask, human_flags, 0).astype(np.float32)
            inputs = tec[0, ...].transpose((1, 2, 0)).reshape((Na, Nt, Nd, 1)).astype(np.float32) / 55.

            ref_directions = np.tile(ref_directions[None, :, :], (self.crop_size, 1, 1))
            directions = np.tile(directions[None, :, :], (self.crop_size, 1, 1))

            cal_pos = np.concatenate([directions, ref_directions], axis=-1)

            ref_dist = np.tile(ref_dist[:, None, :], [1, self.crop_size, 1])  # Na,cropsize, 3

            # buffer
            for b in range(Na):
                for start in range(0, Nt, self.crop_size):
                    stop = start + self.crop_size
                    if stop > Nt:
                        stop = Nt
                        start = stop - self.crop_size
                    if np.sum(mask[b, start:stop, :, :]) == 0:
                        continue
                    yield (inputs[b, start:stop, :, :], labels[b, start:stop, :, :], mask[b, start:stop, :, :], cal_pos,
                           ref_dist[b, :, :])

        return


class Trainer(object):
    # _module = os.path.dirname(sys.modules["bayes_gain_screens"].__file__)
    # flagging_models = os.path.join(_module, 'flagging_models')

    def __init__(self, num_cal=45, num_ant=62, batch_size=16, graph=None, output_bias=0., pos_weight=1., crop_size=60):
        if graph is None:
            graph = tf.Graph()
        self.graph = graph
        self.crop_size = crop_size
        self.processed_per_batch = batch_size
        with self.graph.as_default():
            self.datapacks_pl = tf.placeholder(tf.string, shape=[None], name='datapacks')
            # self.shard_idx = tf.placeholder(tf.int64, shape=[])
            self.training_pl = tf.placeholder(tf.bool, shape=[])

            ###
            # train/test inputs

            dataset = tf.data.Dataset.from_tensors((self.datapacks_pl,))  # .shard(2, self.shard_idx)
            dataset = dataset.interleave(lambda datapacks:
                                         tf.data.Dataset.from_generator(
                                             TrainingDataGen(self.crop_size),
                                             output_types=(tf.float32, tf.float32, tf.float32, tf.float32, tf.float32),
                                             output_shapes=(tf.TensorShape((self.crop_size, num_cal, 1)),
                                                            tf.TensorShape((self.crop_size, num_cal, 1)),
                                                            tf.TensorShape((self.crop_size, num_cal, 1)),
                                                            tf.TensorShape([self.crop_size, num_cal, 4]),
                                                            tf.TensorShape([self.crop_size, 3])),
                                             args=(datapacks,)),
                                         cycle_length=1,
                                         block_length=1
                                         )
            dataset = dataset.shuffle(1000)
            dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)

            iterator_tensor = dataset.make_initializable_iterator()
            self.init = iterator_tensor.initializer
            inputs, labels, mask, cal_pos, ant_pos = iterator_tensor.get_next()

            model = Model(class_bias=output_bias, rate=0.1)

            logits = model(inputs, cal_pos, ant_pos, training=self.training_pl)

            loss = tf.nn.weighted_cross_entropy_with_logits(labels=labels,
                                                            logits=logits,
                                                            pos_weight=pos_weight)
            loss = tf.reduce_mean(loss * mask)

            global_step = tf.Variable(0, trainable=False)
            lr = tf.train.inverse_time_decay(1e-3, global_step, 1000, 0.25)
            optimiser = tf.train.AdamOptimizer(lr)
            opt_op = optimiser.minimize(loss, global_step=global_step, var_list=model.all_trainable_variables)

            self.threshold = tf.Variable(0.5, dtype=tf.float32)
            self.threshold_pl = tf.placeholder(tf.float32, [])
            self.assign_threshold = tf.assign(self.threshold, self.threshold_pl)

            global_conf_mat = tf.Variable(tf.zeros([2, 2], dtype=tf.float32),
                                          trainable=False)
            self.reset_metric = global_conf_mat.initializer

            inst_conf_mat = tf.math.confusion_matrix(tf.cast(tf.reshape(labels, (-1,)), tf.int32),
                                                     tf.nn.sigmoid(tf.reshape(logits, (-1,))) > self.threshold,
                                                     num_classes=2,
                                                     dtype=tf.float32)

            self.global_conf_mat = tf.assign_add(global_conf_mat, inst_conf_mat)

            self.global_metrics = self.multi_class_metrics(self.global_conf_mat)

            self.train_summaries = tf.summary.merge([tf.summary.scalar('loss', loss, family='train'),
                                                     tf.summary.scalar('acc', self.global_metrics['avg_acc'],
                                                                       family='train'),
                                                     tf.summary.scalar('fnr', self.global_metrics['avg_fnr'],
                                                                       family='train'),
                                                     tf.summary.scalar('fpr', self.global_metrics['avg_fpr'],
                                                                       family='train'),
                                                     tf.summary.scalar('learning_rate', lr, family='train'),
                                                     tf.summary.image('conf_mat',
                                                                      self.global_conf_mat[None, :, :, None],
                                                                      family='train')]
                                                    )
            self.test_summaries = tf.summary.merge([tf.summary.scalar('loss', loss, family='test'),
                                                    tf.summary.scalar('acc', self.global_metrics['avg_acc'],
                                                                      family='test'),
                                                    tf.summary.scalar('fnr', self.global_metrics['avg_fnr'],
                                                                      family='test'),
                                                    tf.summary.scalar('fpr', self.global_metrics['avg_fpr'],
                                                                      family='test'),
                                                    tf.summary.image('conf_mat', self.global_conf_mat[None, :, :, None],
                                                                     family='test')
                                                    ])

            eval_features_pl = tf.placeholder(tf.float32, shape=[None, None, None, 1])
            eval_cal_pos_pl = tf.placeholder(tf.float32, shape=[None, 4])
            eval_cal_pos = tf.broadcast_to(eval_cal_pos_pl, tf.broadcast_dynamic_shape(tf.shape(eval_cal_pos_pl),
                                                                                       tf.shape(eval_features_pl)))
            eval_ant_pos_pl = tf.placeholder(tf.float32, shape=[None, 3])
            eval_ant_pos = tf.broadcast_to(eval_ant_pos_pl[:, None, :],
                                           tf.broadcast_dynamic_shape(tf.shape(eval_ant_pos_pl[:, None, :]),
                                                                      tf.shape(eval_features_pl[:, :, 0, :])))
            print(eval_features_pl, eval_cal_pos, eval_ant_pos)
            eval_logits = model(eval_features_pl, eval_cal_pos, eval_ant_pos, training=False)
            eval_prob = tf.nn.sigmoid(eval_logits)
            eval_class = eval_prob > self.threshold

        self.eval_features = eval_features_pl
        self.eval_cal_pos = eval_cal_pos_pl
        self.eval_ant_onehot = eval_ant_pos_pl
        self.eval_prob = eval_prob
        self.eval_class = eval_class
        self.preds = tf.nn.sigmoid(logits)
        self.labels = labels
        self.mask = mask
        self.lr = lr
        self.opt_op = opt_op
        self.loss = loss
        self.global_step = global_step
        self.graph = graph

    def multi_class_metrics(self, conf_mat):
        """
        Compute the multi-class metrics.

        Args:
            conf_mat:

        Returns:

        """
        conf_mat = tf.cast(conf_mat, tf.float32)
        #  P  N
        # T TP FN = Total T
        # F FP TN = Total F
        # How many classifications were right.
        tp = tf.linalg.diag_part(conf_mat)
        # How many classifications were class A that should not have been are the ones in the column of A minus diagonal
        fp = tf.reduce_sum(conf_mat, axis=0) - tp
        # How many classifications were not class A that should have been are the ones in the row of A minus diagonal
        fn = tf.reduce_sum(conf_mat, axis=1) - tp
        # How many classifications were not class A and were not in row or column A
        tn = tf.reduce_sum(conf_mat) - tp - fp - fn
        total = tp + fp + fn + tn
        # correct / total
        acc = (tp + tn) / total
        avg_acc = tf.reduce_mean(acc)
        fpr = fp / (fp + tn)
        avg_fpr = tf.reduce_mean(fpr)
        fnr = fn / (fn + tp)
        avg_fnr = tf.reduce_mean(fnr)

        results = dict(acc=acc, fpr=fpr, fnr=fnr, avg_acc=avg_acc, avg_fnr=avg_fnr, avg_fpr=avg_fpr)

        C = tf.reduce_mean(conf_mat, axis=1)
        num_classes = tf.size(C)
        random_conf_mat = (C[:, None] * tf.ones([num_classes], dtype=C.dtype))
        tp = tf.linalg.diag_part(random_conf_mat)
        fp = tf.reduce_sum(random_conf_mat, axis=0) - tp
        fn = tf.reduce_sum(random_conf_mat, axis=1) - tp
        tn = tf.reduce_sum(random_conf_mat) - tp - fp - fn
        total = tp + fp + fn + tn
        random_acc = (tp + tn) / total
        random_avg_acc = tf.reduce_mean(random_acc)
        random_fpr = fp / (fp + tn)
        random_fnr = fn / (fn + tp)
        random_avg_fpr = tf.reduce_mean(random_fpr)
        random_avg_fnr = tf.reduce_mean(random_fnr)

        results['rel_acc'] = acc / random_acc
        results['rel_avg_acc'] = avg_acc / random_avg_acc
        results['rel_avg_fpr'] = avg_fpr / random_avg_fpr
        results['rel_avg_fnr'] = avg_fnr / random_avg_fnr
        results['rel_fpr'] = fpr / random_fpr
        results['rel_fnr'] = fnr / random_fnr

        return results

    def export_model(self, sess: tf.Session, model_dir, version):
        with sess.graph.as_default():
            export_path = os.path.join(model_dir, 'model', str(version))
            logger.info('Exporting trained model to {}'.format(export_path))
            builder = tf.saved_model.builder.SavedModelBuilder(export_path)
            # the inputs and outputs
            tensor_info_input = tf.saved_model.utils.build_tensor_info(self.eval_features)
            tensor_info_input_pos = tf.saved_model.utils.build_tensor_info(self.eval_cal_pos)
            tensor_info_input_ant_onehot = tf.saved_model.utils.build_tensor_info(self.eval_ant_onehot)
            tensor_info_output_class = tf.saved_model.utils.build_tensor_info(self.eval_class)
            tensor_info_output_prob = tf.saved_model.utils.build_tensor_info(self.eval_prob)
            # signature made using util
            prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
                inputs={'tec': tensor_info_input,
                        'pos': tensor_info_input_pos,
                        'ant_pos': tensor_info_input_ant_onehot
                        },
                outputs={'probability': tensor_info_output_prob,
                         'class': tensor_info_output_class
                         },
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

    def train(self, datapacks, epochs, training_dir, model_dir, version, print_freq=10):
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(training_dir, exist_ok=True)
        logdir = os.path.join(training_dir, 'logs')

        with tf.Session(graph=self.graph, config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            with tf.summary.FileWriter(logdir, sess.graph, session=sess) as writer:
                saver = tf.train.Saver()
                print("Initialising valiables")
                sess.run(tf.initialize_variables(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)))
                print("Restoring if possibe")
                try:
                    saver.restore(sess, tf.train.latest_checkpoint(training_dir))
                except:
                    print("No model found to restore")
                    pass
                for epoch in range(epochs):
                    sess.run([self.init,  # reset data feed
                              self.reset_metric],  # reset global conf matrix
                             {
                                 self.datapacks_pl: datapacks[::2],
                                 # self.shard_idx:0
                             })
                    train_loss = 0.
                    batch = 0
                    # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    # run_metadata = tf.RunMetadata()
                    t0 = default_timer()
                    while True:
                        # extra_kwargs = dict(options=run_options, run_metadata=run_metadata)
                        try:
                            _, global_step, lr, loss, train_conf_mat, summaries, train_global_metrics = sess.run(
                                [self.opt_op, self.global_step, self.lr,
                                 self.loss, self.global_conf_mat,
                                 self.train_summaries, self.global_metrics],
                                {self.training_pl: True}
                            )
                            train_loss = train_loss + loss
                            batch += 1
                            if global_step % print_freq == 0:
                                writer.add_summary(summaries, global_step)
                                rate = self.processed_per_batch * print_freq / (default_timer() - t0)
                                t0 = default_timer()
                                print(
                                    "Epoch {:02d} Step {:04d} [{:.1f} / second] Train loss {:.5f} Learning rate {:.5f} Avg. Acc. {:.5f} Avg. FPR {:.5f} Avg. FNR {:.5f}".format(
                                        epoch,
                                        global_step,
                                        rate,
                                        loss,
                                        lr, train_global_metrics['avg_acc'], train_global_metrics['avg_fpr'],
                                        train_global_metrics['avg_fnr']))
                                # print('Saving...')
                                save_path = saver.save(sess, os.path.join(training_dir, "model"),
                                                       global_step=self.global_step)
                                # print("Saved to {}".format(save_path))

                        except tf.errors.OutOfRangeError:
                            break

                    sess.run([self.init,  # reset data feed
                              self.reset_metric],  # reset global conf matrix
                             {
                                 self.datapacks_pl: datapacks[1::2],
                                 # self.shard_idx: 1
                             })
                    test_loss = 0.
                    batch = 0
                    test_preds, test_labels, test_masks = [], [], []
                    # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    # run_metadata = tf.RunMetadata()
                    while True:
                        try:
                            global_step, lr, loss, test_conf_mat, summaries, test_global_metrics, probs, labels, mask = sess.run(
                                [self.global_step, self.lr, self.loss, self.global_conf_mat, self.test_summaries,
                                 self.global_metrics, self.preds, self.labels, self.mask],
                                {self.training_pl: False},
                                # options=run_options,
                                # run_metadata=run_metadata
                            )
                            test_preds.append(probs)
                            test_labels.append(labels)
                            test_masks.append(mask)
                            test_loss = test_loss + loss
                            batch += 1
                        except tf.errors.OutOfRangeError:
                            break

                    test_preds = np.concatenate(test_preds, axis=0)
                    test_labels = np.concatenate(test_labels, axis=0)
                    test_masks = np.concatenate(test_masks, axis=0)

                    fpr, tpr, thresholds = roc_curve(test_labels.flatten(), test_preds.flatten(),
                                                     sample_weight=test_masks.flatten())
                    which = np.argmax(tpr - fpr)
                    threshold = thresholds[which]

                    plt.plot(fpr, tpr, c='black')
                    plt.scatter(fpr[which], tpr[which], c='red', label='opt')
                    for t in np.arange(1, 10) / 10.:
                        _which = np.argmin(np.abs(thresholds - t))
                        plt.scatter(fpr[_which], tpr[_which], label='{:.1f}'.format(t))
                    plt.legend()
                    plt.savefig("./thresholds.png")
                    plt.close('all')

                    print("New optimal threshold: {}".format(threshold))
                    # sess.run(self.assign_threshold, {self.threshold_pl: threshold})

                    # writer.add_run_metadata(run_metadata, 'step{:03d}'.format(global_step))
                    writer.add_summary(summaries, global_step)

                    print("Train Results: Epoch {:02d} Train loss {:.5f}\nconf. mat.\n{}".format(epoch,
                                                                                                 train_loss / batch,
                                                                                                 train_conf_mat.astype(
                                                                                                     np.int32)
                                                                                                 ))
                    for k, v in train_global_metrics.items():
                        print("TRAIN\t{}:{}".format(k, v))

                    print("Test Results: Epoch {:02d} Test loss {:.5f}\nconf. mat.\n{}".format(epoch,
                                                                                               test_loss / batch,
                                                                                               test_conf_mat.astype(
                                                                                                   np.int32)))
                    for k, v in test_global_metrics.items():
                        print("TEST\t{}:{}".format(k, v))

            # save model
            print("Exporting model...")
            self.export_model(sess, model_dir, version)


def get_output_bias(datapacks):
    num_pos = 0
    num_neg = 0
    for datapack in datapacks:
        human_flags = np.load(datapack)['human_flags']
        num_pos += np.sum(human_flags == 1)
        num_neg += np.sum(human_flags == 0)
    pos_weight = num_neg / num_pos
    bias = np.log(num_pos) - np.log(num_neg)
    return bias, pos_weight


def main(data_dir, epochs, batch_size, crop_size, model_dir, training_dir, version):
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
    data_dir = os.path.abspath(data_dir)
    datapacks = sorted(glob.glob(os.path.join(data_dir, '*.joint.npz')))
    # log
    output_bias, pos_weight = get_output_bias(datapacks)

    num_cal = np.load(datapacks[0])['tec'].shape[1]

    trainer = Trainer(num_cal=num_cal, batch_size=batch_size, graph=None, output_bias=output_bias,
                      pos_weight=pos_weight, crop_size=crop_size)
    trainer.train(datapacks, epochs, training_dir, model_dir, version, print_freq=10)


def make_fake_data(test_dir):
    tec = np.random.normal(size=(1, 45, 6, 100))
    directions = np.random.normal(size=(45, 2))
    human_flags = np.random.randint(-1, 2, size=(45, 6, 100))
    human_flags[tec[0, ...] > 0.] = 1
    human_flags[tec[0, ...] < 0.] = 0
    human_flags[tec[0, ...] < -0.5] = -1
    os.makedirs(test_dir, exist_ok=True)
    datapack = os.path.join(test_dir, 'test_data.joint.npz')
    np.savez(datapack, human_flags=human_flags, tec=tec, directions=directions)

    return datapack


def add_args(parser):
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument('--epochs', help='Number of epochs to train for', default=20, type=int, required=False)
    parser.add_argument('--batch_size', help='Number of epochs to train for', default=16, type=int, required=False)
    parser.add_argument('--crop_size', help='length of tec series to train on at a time', default=50, type=int,
                        required=False)
    parser.add_argument('--model_dir', help='Model save directory.', default='models', type=str, required=False)
    parser.add_argument('--data_dir', help='Where data is.', type=str, required=True)
    parser.add_argument('--training_dir', help='Training and logs directory.', default='training', type=str,
                        required=False)
    parser.add_argument('--version', help='Version of model to save as.', default=6, type=int, required=False)


def test_main():
    make_fake_data('../test_dir')
    main(data_dir='../test_dir', epochs=1, batch_size=2, crop_size=10,
         model_dir='../models', training_dir='../training', version=1)


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