import os

from .common_setup import *
import numpy as np
import os
import tensorflow as tf

from bayes_filter import jitter
from bayes_filter.misc import (random_sample, flatten_batch_dims, load_array_file, timer, diagonal_jitter,
    log_normal_solve_fwhm,make_example_datapack, maybe_create_posterior_solsets, graph_store_get,graph_store_set,
                               safe_cholesky, get_screen_directions, lock_print)

def test_getset_graph_ops(tf_graph):
    a = tf.constant(0)
    graph_store_set('a',a, graph=tf_graph)
    a_out = graph_store_get('a',tf_graph)
    assert a == a_out

def test_get_screen_directions():
    get_screen_directions('/home/albert/ftp/image.pybdsm.srl.fits')


def test_random_sample(tf_session):
    with tf_session.graph.as_default():
        t = tf.ones((6,5,8))
        assert tf_session.run(random_sample(t)).shape == (6,5,8)
        assert tf_session.run(random_sample(t,3)).shape == (3, 5, 8)
        assert tf_session.run(random_sample(t, 9)).shape == (6, 5, 8)

def test_lock_print(tf_session):
    with tf_session.graph.as_default():
        t = tf.constant(True)
        with tf.control_dependencies([lock_print(t, "This is a message")]):
            t = tf.constant(True)
            assert tf_session.run(t)



def test_safe_cholesky(tf_session):
    with tf_session.graph.as_default():
        A = tf.ones((5, 6,6), float_type)
        K = tf.matmul(A, A, transpose_b=True)
        L = safe_cholesky(K)
        assert tf_session.run(L).shape == (5,6,6)

def test_flatten_batch_dims(tf_session):
    with tf_session.graph.as_default():
        t = tf.ones((1,2,3,4))
        f = flatten_batch_dims(t)
        assert tuple(tf_session.run(tf.shape(f))) == (6,4)

        t = tf.ones((2,1, 2, 3, 4))
        f = flatten_batch_dims(t,num_batch_dims=3)
        assert tuple(tf_session.run(tf.shape(f))) == (4, 3 , 4)


def test_load_array_file(arrays):
    lofar_array = os.path.join(arrays, 'arrays/lofar.hba.antenna.cfg')
    lofar_cycle0_array = os.path.join(arrays, 'arrays/lofar.cycle0.hba.antenna.cfg')
    gmrt_array = os.path.join(arrays, 'arrays/gmrtPos.csv')
    lofar_array = load_array_file(lofar_array)
    lofar_cycle0_array = load_array_file(lofar_cycle0_array)
    gmrt_array = load_array_file(gmrt_array)
    assert (len(lofar_array[0]),3) == lofar_array[1].shape
    assert (len(lofar_cycle0_array[0]), 3) == lofar_cycle0_array[1].shape
    assert (len(gmrt_array[0]), 3) == gmrt_array[1].shape


def test_timer(tf_session):
    with tf_session.graph.as_default():
        t0 = timer()
        with tf.control_dependencies([t0]):
            t1 = timer()
        t0, t1 = tf_session.run([t0,t1])
        assert t1 > t0


def test_diagonal_jitter(tf_session):
    with tf_session.graph.as_default():
        j = diagonal_jitter(5)
        assert np.all(tf_session.run(j) == jitter*np.eye(5))


def test_log_normal_solve_fwhm():
    mu, stddev = log_normal_solve_fwhm(np.exp(1), np.exp(2), np.exp(-1))
    assert np.isclose(stddev**2, 0.5 * (0.5) ** 2)
    assert np.isclose(mu, 3./2. + 0.5 * (0.5) ** 2)

def test_make_example_datapack():
    res = make_example_datapack(4,2,2,obs_type='DDTEC',name=os.path.join(TEST_FOLDER,'test_misc.h5'),clobber=True, return_full=True)
    datapack = res['datapack']
    datapack.current_solset = 'sol000'
    assert 'sol000' in datapack.solsets
    assert 'phase000' in datapack.soltabs
    assert 'tec000' in datapack.soltabs
    assert np.any(np.abs(datapack.phase[0])>0)

    assert np.all(np.isclose(res['phase'], datapack.phase[0]))
    assert np.all(np.isclose(res['dtec'], datapack.tec[0]))
    # assert np.all(np.isclose(res['antennas'], datapack.get_antennas(datapack.antennas[0])[1]))
    # assert np.all(np.isclose(res['directions'], datapack.get_directions(datapack.directions[0])[1]))


def test_maybe_create_posterior_solsets():
    datapack = make_example_datapack(4,2,3,name=os.path.join(TEST_FOLDER,'test_misc.h5'),clobber=True)
    patch_names, _ = datapack.directions
    _, screen_directions = datapack.get_directions(patch_names)
    maybe_create_posterior_solsets(datapack, 'sol000', posterior_name='posterior',screen_directions=screen_directions, remake_posterior_solsets=False)
    assert "data_posterior" in datapack.solsets
    assert "screen_posterior" in datapack.solsets
    maybe_create_posterior_solsets(datapack, 'sol000', posterior_name='posterior',screen_directions=screen_directions, remake_posterior_solsets=True)
    assert "data_posterior" in datapack.solsets
    assert "screen_posterior" in datapack.solsets
    print(datapack)