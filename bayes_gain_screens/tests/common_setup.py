import pytest
import tensorflow as tf

@pytest.fixture
def tf_graph():
    return tf.Graph()


@pytest.fixture
def tf_session(tf_graph):
    sess = tf.Session(graph=tf_graph)
    return sess
