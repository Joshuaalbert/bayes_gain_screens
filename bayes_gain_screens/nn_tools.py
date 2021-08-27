import tensorflow as tf
from graph_nets.graphs import GraphsTuple
from graph_nets import utils_tf, blocks
import tqdm
import sonnet as snt
from sonnet.src.base import Optimizer, Module
import numpy as np
import six
import abc
import contextlib
from typing import List
import os

@six.add_metaclass(abc.ABCMeta)
class AbstractModule(snt.Module):
    """Makes Sonnet1-style childs from this look like a Sonnet2 module."""
    def __init__(self, *args, **kwargs):
        super(AbstractModule, self).__init__(*args, **kwargs)
        self.__call__.__func__.__doc__ = self._build.__doc__  # pytype: disable=attribute-error

    # In snt2 calls to `_enter_variable_scope` are ignored.
    @contextlib.contextmanager
    def _enter_variable_scope(self, *args, **kwargs):
        yield None

    def __call__(self, *args, **kwargs):
        return self._build(*args, **kwargs)

    @abc.abstractmethod
    def _build(self, *args, **kwargs):
        """Similar to Sonnet 1 ._build method."""

class TrainOneEpoch(Module):
    _model:AbstractModule
    _opt:Optimizer

    def __init__(self, model:AbstractModule, loss, opt:Optimizer, strategy:tf.distribute.MirroredStrategy=None, name=None):
        super(TrainOneEpoch, self).__init__(name=name)
        self.epoch = tf.Variable(0, dtype=tf.int64)
        self.minibatch = tf.Variable(0, dtype=tf.int64)
        self._model = model
        self._model.step = self.minibatch
        self._opt = opt
        self._loss = loss
        self._strategy = strategy
        self._checkpoint = tf.train.Checkpoint(module=model)


    @property
    def strategy(self) -> tf.distribute.MirroredStrategy:
        return self._strategy

    @property
    def model(self):
        return self._model

    @property
    def opt(self):
        return self._opt

    def loss(self, model_output, batch):
        return self._loss(model_output, batch)

    def train_step(self, batch):
        """
        Trains on a single batch.

        Args:
            batch: user defined batch from a dataset.

        Returns:
            loss
        """
        with tf.GradientTape() as tape:
            model_output = self.model(batch)
            loss = self.loss(model_output, batch)
        params = self.model.trainable_variables
        grads = tape.gradient(loss, params)

        if self.strategy is not None:
            replica_ctx = tf.distribute.get_replica_context()
            grads = replica_ctx.all_reduce("mean", grads)
        for (param, grad) in zip(params, grads):
            if grad is not None:
                tf.summary.histogram(param.name+"_grad",grad, step=self.minibatch)
        self.opt.apply(grads, params)
        return loss

    def one_epoch_step(self, train_dataset):
        """
        Updates a model with one epoch of train_one_epoch, and returns a dictionary of values to monitor, i.e. metrics.

        Returns:
            average loss
        """
        self.epoch.assign_add(1)
        # metrics = None
        loss = 0.
        num_batches = 0.
        if self.strategy is not None:
            train_dataset = self.strategy.experimental_distribute_dataset(train_dataset)
        for train_batch in train_dataset:
            self.minibatch.assign_add(1)
            if self.strategy is not None:
                _loss = self.strategy.run(self.train_step, args=(train_batch,))
                _loss = self.strategy.reduce("sum", _loss, axis=None)
            else:
                _loss = self.train_step(train_batch)
            tf.summary.scalar('mini_batch_loss',_loss, step=self.minibatch)
            loss += _loss
            num_batches += 1.
        tf.summary.scalar('epoch_loss', loss/num_batches, step=self.epoch)
        return loss/num_batches

    def evaluate(self, test_dataset):
        loss = 0.
        num_batches = 0.
        if self.strategy is not None:
            test_dataset = self.strategy.experimental_distribute_dataset(test_dataset)
        for test_batch in test_dataset:
            if self.strategy is not None:
                model_output = self.strategy.run(self.model, args=(test_batch,))
                _loss = self.strategy.run(self.loss, args=(model_output, test_batch))
                loss += self.strategy.reduce("sum", _loss, axis=0)
            else:
                model_output = self.model(test_batch)
                loss += self.loss(model_output, test_batch)
            num_batches += 1.
        tf.summary.scalar('loss', loss / num_batches, step=self.epoch)
        return loss / num_batches


def get_distribution_strategy(use_cpus=True, logical_per_physical_factor=1, memory_limit=2000) -> tf.distribute.MirroredStrategy:
    # trying to set GPU distribution
    physical_gpus = tf.config.experimental.list_physical_devices("GPU")
    physical_cpus = tf.config.experimental.list_physical_devices("CPU")
    if len(physical_gpus) > 0 and not use_cpus:
        print("Physical GPUS: {}".format(physical_gpus))
        if logical_per_physical_factor > 1:
            for dev in physical_gpus:
                tf.config.experimental.set_virtual_device_configuration(
                    dev,
                    [tf.config.experimental.VirtualDeviceConfiguration(
                        memory_limit=memory_limit)] * logical_per_physical_factor
                )

        gpus = tf.config.experimental.list_logical_devices("GPU")

        print("Logical GPUs: {}".format(gpus))

        strategy = snt.distribute.Replicator(
            ["/device:GPU:{}".format(i) for i in range(len(gpus))],
            tf.distribute.ReductionToOneDevice("GPU:0"))
    else:
        print("Physical CPUS: {}".format(physical_cpus))
        if logical_per_physical_factor > 1:
            for dev in physical_cpus:
                tf.config.experimental.set_virtual_device_configuration(
                    dev,
                    [tf.config.experimental.VirtualDeviceConfiguration()] * logical_per_physical_factor
                )

        cpus = tf.config.experimental.list_logical_devices("CPU")
        print("Logical CPUs: {}".format(cpus))

        strategy = snt.distribute.Replicator(
            ["/device:CPU:{}".format(i) for i in range(len(cpus))],
            tf.distribute.ReductionToOneDevice("CPU:0"))

    return strategy

def _round(v, last_v):
    if last_v is None:
        uncert_v = v
    else:
        uncert_v = abs(v - last_v)
    sig_figs = -int("{:e}".format(uncert_v).split('e')[1]) + 1

    return round(float(v), sig_figs)

def vanilla_training_loop(train_one_epoch:TrainOneEpoch, training_dataset, test_dataset=None, num_epochs=1,
                          early_stop_patience=None, checkpoint_dir=None, log_dir=None, debug=False):
    """
    Does simple training.

    Args:
        training_dataset: Dataset for training
        train_one_epoch: TrainOneEpoch
        num_epochs: how many epochs to train
        test_dataset: Dataset for testing
        early_stop_patience: Stops training after this many epochs where test dataset loss doesn't improve
        checkpoint_dir: where to save epoch results.
        debug: bool, whether to use debug mode.

    Returns:

    """
    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir,exist_ok=True)

    training_dataset = training_dataset.prefetch(tf.data.experimental.AUTOTUNE)#.cache()
    if test_dataset is not None:
        test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)#.cache()

    # We'll turn the one_epoch_step function which updates our models into a tf.function using
    # autograph. This makes train_one_epoch much faster. If debugging, you can turn this
    # off by setting `debug = True`.
    step = train_one_epoch.one_epoch_step
    evaluate = train_one_epoch.evaluate
    if not debug:
        step = tf.function(step)
        evaluate = tf.function(evaluate)

    fancy_progress_bar = tqdm.tqdm(range(num_epochs),
                                    unit='epochs',
                                    position=0)
    early_stop_min_loss = np.inf
    early_stop_interval = 0

    train_log_dir = os.path.join(log_dir,"train")
    test_log_dir = os.path.join(log_dir,"test")
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    checkpoint = tf.train.Checkpoint(module=train_one_epoch)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3,
                                         checkpoint_name=train_one_epoch.model.__class__.__name__)
    if manager.latest_checkpoint is not None:
        checkpoint.restore(manager.latest_checkpoint)
        print(f"Restored from {manager.latest_checkpoint}")
    last_loss = None
    last_test_loss = None
    for step_num in fancy_progress_bar:
        with train_summary_writer.as_default():
            loss = step(iter(training_dataset))
        tqdm.tqdm.write(
            '\nEpoch = {}/{} (loss = {})'.format(
                train_one_epoch.epoch.numpy(), num_epochs, _round(loss,last_loss)))
        last_loss = loss
        if test_dataset is not None:
            with test_summary_writer.as_default():
                test_loss = evaluate(iter(test_dataset))
            tqdm.tqdm.write(
                '\n\t(Test loss = {})'.format(_round(test_loss,last_test_loss)))
            last_test_loss = test_loss
            if early_stop_patience is not None:
                if test_loss <= early_stop_min_loss:
                    early_stop_min_loss = test_loss
                    early_stop_interval = 0
                    manager.save()
                else:
                    early_stop_interval += 1
                if early_stop_interval == early_stop_patience:
                    tqdm.tqdm.write(
                        '\n\tStopping Early')
                    break
            else:
                manager.save()
        else:
            manager.save()
    train_summary_writer.close()
    test_summary_writer.close()