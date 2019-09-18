import os
import sys

from .. import logging
import numpy as np
import pytest
import tensorflow as tf

from bayes_filter import float_type
from bayes_filter.coord_transforms import tf_coord_transform, itrs_to_enu_6D
from bayes_filter.feeds import IndexFeed, TimeFeed, CoordinateFeed, DataFeed

from bayes_filter.misc import load_array_file

TEST_FOLDER = os.path.abspath('./test_output')
os.makedirs(TEST_FOLDER,exist_ok=True)

def clean_test_output():
    logging.debug("Removing {}".format(TEST_FOLDER))
    os.unlink(TEST_FOLDER)



@pytest.fixture
def tf_graph():
    return tf.Graph()


@pytest.fixture
def tf_session(tf_graph):
    sess = tf.Session(graph=tf_graph)
    return sess


@pytest.fixture
def arrays():
    return os.path.dirname(sys.modules["bayes_filter"].__file__)


@pytest.fixture
def lofar_array(arrays):
    lofar_array = os.path.join(arrays, 'arrays/lofar.hba.antenna.cfg')
    return load_array_file(lofar_array)


@pytest.fixture
def lofar_array2(arrays):
    lofar_array = os.path.join(arrays, 'arrays/lofar.hba.antenna.cfg')
    res = load_array_file(lofar_array)
    return res[0][::2], res[1][::2,:]


@pytest.fixture
def index_feed(tf_graph):
    with tf_graph.as_default():
        return IndexFeed(2)


@pytest.fixture
def time_feed(tf_graph, index_feed):
    with tf_graph.as_default():
        times = tf.linspace(0.,100.,9)[:,None]
        return TimeFeed(index_feed,times)


@pytest.fixture
def coord_feed(tf_graph, time_feed, lofar_array):
    with tf_graph.as_default():
        ra = np.pi/4 + 2.*np.pi/180. * tf.random_normal(shape=(4,1))
        dec = np.pi / 4 + 2. * np.pi / 180. * tf.random_normal(shape=(4, 1))
        Xd = tf.concat([ra,dec],axis=1)
        Xa = tf.constant(lofar_array[1],dtype=float_type)
        return CoordinateFeed(time_feed, Xd, Xa, coord_map = tf_coord_transform(itrs_to_enu_6D))


@pytest.fixture
def data_feed(tf_graph, index_feed):
    with tf_graph.as_default():
        shape1 = (1,2,3,4)
        shape2 = (1, 2, 3, 4)
        data1 = tf.ones(shape1)
        data2 = tf.ones(shape2)
        return DataFeed(index_feed, data1, data2)