from .common_setup import *

import numpy as np
import tensorflow as tf
from astropy import time as at

from bayes_filter import float_type
from bayes_filter.coord_transforms import itrs_to_enu_6D, tf_coord_transform, itrs_to_enu_with_references, ITRSToENUWithReferences
from bayes_filter.feeds import IndexFeed, TimeFeed, CoordinateFeed, init_feed
from bayes_filter.misc import make_coord_array


def test_itrs_to_enu_6D(tf_session, time_feed, lofar_array):
    # python test
    times = np.arange(2)[:,None]
    directions = np.random.normal(0,0.1, size=(10,2))
    antennas = lofar_array[1]
    X = make_coord_array(times, directions, antennas,flat=False)
    out = np.array(list(map(itrs_to_enu_6D,X)))
    assert out.shape == (2,10,antennas.shape[0],7)
    #TF test
    with tf_session.graph.as_default():
        index_feed = IndexFeed(1)
        obstime_init = at.Time("2018-01-01T00:00:00.000", format='isot')
        times = tf.linspace(obstime_init.mjd * 86400., obstime_init.mjd * 86400. + 100., 9)[:, None]
        time_feed = TimeFeed(index_feed, times)
        ra = np.pi / 4 + 2. * np.pi / 180. * tf.random_normal(shape=(4, 1))
        dec = np.pi / 4 + 2. * np.pi / 180. * tf.random_normal(shape=(4, 1))
        Xd = tf.concat([ra, dec], axis=1)
        Xa = tf.constant(lofar_array[1], dtype=float_type)
        coord_feed = CoordinateFeed(time_feed, Xd, Xa, coord_map=tf_coord_transform(itrs_to_enu_6D))
        init, next = init_feed(coord_feed)
        tf_session.run(init)
        out, N, slice_size = tf_session.run([next, coord_feed.N, coord_feed.slice_size])
        assert out.shape[0] == slice_size * 4 * len(lofar_array[0])
        assert out.shape[1] == 7
        assert np.all(np.isclose(np.linalg.norm(out[:,1:4],axis=1), 1.))
        assert np.all(np.isclose(np.linalg.norm(out[:, 4:7], axis=1) < 100., 1.))


def test_itrs_to_enu_with_references(tf_session, time_feed, lofar_array):
    # python test
    times = np.arange(2)[:,None]
    directions = np.random.normal(0,0.1, size=(10,2))
    antennas = lofar_array[1]
    X = make_coord_array(times, directions, antennas,flat=False)
    out = np.array(list(map(itrs_to_enu_6D,X)))
    assert out.shape == (2,10,antennas.shape[0],7)
    #TF test
    with tf_session.graph.as_default():
        index_feed = IndexFeed(1)
        obstime_init = at.Time("2018-01-01T00:00:00.000", format='isot')
        times = tf.linspace(obstime_init.mjd * 86400., obstime_init.mjd * 86400. + 100., 9)[:, None]
        time_feed = TimeFeed(index_feed, times)
        ra = np.pi / 4 + 2. * np.pi / 180. * tf.random_normal(shape=(4, 1))
        dec = np.pi / 4 + 2. * np.pi / 180. * tf.random_normal(shape=(4, 1))
        Xd = tf.concat([ra, dec], axis=1)
        Xa = tf.constant(lofar_array[1], dtype=float_type)
        coord_feed = CoordinateFeed(time_feed, Xd, Xa, coord_map=ITRSToENUWithReferences(lofar_array[1][0,:], [np.pi/4,np.pi/4], lofar_array[1][0,:]))
        init, next = init_feed(coord_feed)
        tf_session.run(init)
        out, N, slice_size = tf_session.run([next, coord_feed.N, coord_feed.slice_size])

        assert np.all(np.isclose(out[0, 4:7], np.zeros_like(out[0,4:7])))
        assert out.shape[0] == slice_size * 4 * len(lofar_array[0])
        assert out.shape[1] == 13
        assert np.all(np.isclose(np.linalg.norm(out[:,1:4],axis=1), 1.))
        assert np.all(np.isclose(np.linalg.norm(out[:, 4:7], axis=1) < 100., 1.))
        assert np.all(np.isclose(np.linalg.norm(out[:, 10:13], axis=1), 1.))
        assert np.all(np.isclose(np.linalg.norm(out[:, 7:10], axis=1) < 100., 1.))