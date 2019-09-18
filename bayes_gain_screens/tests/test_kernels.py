from .common_setup import *

import numpy as np
import pylab as plt
import tensorflow as tf
from astropy import time as at

from bayes_filter import float_type, KERNEL_SCALE
from bayes_filter.coord_transforms import tf_coord_transform, itrs_to_enu_with_references, ITRSToENUWithReferences
from bayes_filter.feeds import IndexFeed, TimeFeed, CoordinateFeed, init_feed, CoordinateDimFeed
from bayes_filter.kernels import Histogram, DTECIsotropicTimeGeneral, DTECIsotropicTimeGeneralODE
from bayes_filter.misc import safe_cholesky


def test_histogram(tf_session):
    with tf_session.graph.as_default():
        heights = tf.constant(np.arange(50)+1,dtype=float_type)
        edgescales = tf.constant(np.arange(51)/50., dtype=float_type)
        heights = tf.exp(-0.5*edgescales**2/0.1**2)[:-1]
        kern = Histogram(heights, edgescales=edgescales)
        x = tf.constant(np.linspace(0.,10.,100)[:,None],dtype=float_type)

        h,e = tf_session.run([kern.heights, kern.edgescales])

        for i in range(50):
            plt.bar(0.5*(e[i+1]+e[i]),h[i],e[i+1]-e[i])
        plt.savefig(os.path.join(TEST_FOLDER,"test_histogram_spectrum.png"))
        plt.close('all')

        K = kern.matrix(x,x)
        K,L = tf_session.run([K,safe_cholesky(K)])
        plt.imshow(K)
        plt.colorbar()
        plt.savefig(os.path.join(TEST_FOLDER,"test_histogram.png"))
        plt.show()#close("all")
        y = np.dot(L, np.random.normal(size=(100,3)))
        plt.plot(np.linspace(0,10.,100),y)
        plt.savefig(os.path.join(TEST_FOLDER,"test_histogram_sample.png"))
        plt.close("all")

        x0 = tf.constant([[0.]],dtype=float_type)
        K_line = kern.matrix(x, x0)
        K_line = tf_session.run(K_line)
        plt.plot(np.linspace(0,10,100),K_line[:,0])
        plt.savefig(os.path.join(TEST_FOLDER,"test_histogram_kline.png"))
        plt.close('all')

def test_isotropic_time_general(tf_session, lofar_array):
    output_folder = os.path.abspath(os.path.join(TEST_FOLDER,'test_kernels'))
    os.makedirs(output_folder,exist_ok=True)
    with tf_session.graph.as_default():
        index_feed = IndexFeed(1)
        obstime_init = at.Time("2018-01-01T00:00:00.000", format='isot')
        times = obstime_init.mjd*86400. + tf.cast(tf.linspace(0., 50., 9)[:, None],float_type)
        time_feed = TimeFeed(index_feed, times)
        ra = np.pi / 4 + 2. * np.pi / 180. * tf.random_normal(shape=(3, 1))
        dec = np.pi / 4 + 2. * np.pi / 180. * tf.random_normal(shape=(3, 1))
        Xd = tf.concat([ra, dec], axis=1)
        Xa = tf.constant(np.concatenate([lofar_array[1][0:1,:], lofar_array[1]],axis=0), dtype=float_type)
        coord_feed = CoordinateFeed(time_feed, Xd, Xa,
                                    coord_map=tf_coord_transform(itrs_to_enu_with_references(lofar_array[1][0,:], [np.pi/4,np.pi/4], lofar_array[1][0,:])))
        init, next = init_feed(coord_feed)

        # index_feed = IndexFeed(2)
        # obstime_init = at.Time("2018-01-01T00:00:00.000", format='isot')
        # times = obstime_init.mjd * 86400. + tf.cast(tf.linspace(0., 50., 9)[:, None], float_type)
        # time_feed = TimeFeed(index_feed, times)
        # ra = np.pi / 4 + 2. * np.pi / 180. * tf.random_normal(shape=(3, 1))
        # dec = np.pi / 4 + 2. * np.pi / 180. * tf.random_normal(shape=(3, 1))
        # Xd = tf.concat([ra, dec], axis=1)
        # Xa = tf.constant(np.concatenate([lofar_array[1][0:1, :], lofar_array[1]], axis=0), dtype=float_type)
        # coord_feed2 = CoordinateFeed(time_feed, Xd, Xa, coord_map=tf_coord_transform(itrs_to_enu_with_references(lofar_array[1][0,:], [np.pi/4,np.pi/4], lofar_array[1][0,:])))
        # dim_feed = CoordinateDimFeed(coord_feed)
        # init2, next2 = init_feed(coord_feed2)
        # init3, next3 = init_feed(dim_feed)

        kern = DTECIsotropicTimeGeneral(variance=4., lengthscales=10.0,
                 a=250., b=50.,  timescale=30., ref_location=[0.,0.,0.],
                                        fed_kernel='RBF',obs_type='DDTEC',kernel_params=dict(resolution=3), squeeze=True)

        K = kern.K(next)
        L = safe_cholesky(K)

        kern2 = DTECIsotropicTimeGeneral(variance=1., lengthscales=10.0,
                                        a=250., b=50., timescale=30., ref_location=[0., 0., 0.],
                                        fed_kernel='RBF', obs_type='DDTEC', kernel_params=dict(resolution=3),
                                        squeeze=True)

        K2 = kern2.K(next)
        L2 = 2*safe_cholesky(K2)

        kern3 = DTECIsotropicTimeGeneral(variance=1.*np.ones((2,1)), lengthscales=10.0*np.ones((2,1)),
                                        a=250.*np.ones((2,1)), b=50.*np.ones((2,1)), timescale=30.*np.ones((2,1)), ref_location=[0., 0., 0.],
                                        fed_kernel='RBF', obs_type='DDTEC', kernel_params=dict(resolution=3),
                                        squeeze=True)

        K3 = kern3.K(next)

        tf_session.run([init])
        X, K, L, L2, K2, K3 = tf_session.run([next, K, L,L2, K2, K3])
        assert np.all(np.isclose(L, L2))
        assert K3.shape == (2,)+K2.shape
        assert np.all(np.isclose(K2,K3))
        assert np.all(K2==K3)
        # import pylab as plt
        # plt.imshow(K)
        # plt.colorbar()
        # plt.savefig(os.path.join(output_folder,'K.png'))
        # plt.close('all')
        #
        # plt.plot(np.diag(K))
        # plt.savefig(os.path.join(output_folder, 'diag.png'))
        # plt.close('all')
        #
        # plt.imshow(L)
        # plt.colorbar()
        # plt.savefig(os.path.join(output_folder, 'L.png'))
        # plt.close('all')


def test_isotropic_time_general_ode(tf_session, lofar_array):
    with tf_session.graph.as_default():
        index_feed = IndexFeed(1)
        obstime_init = at.Time("2018-01-01T00:00:00.000", format='isot')
        times = obstime_init.mjd*86400. + tf.cast(tf.linspace(0., 50., 9)[:, None],float_type)
        time_feed = TimeFeed(index_feed, times)
        ra = np.pi / 4 + 2. * np.pi / 180. * tf.random_normal(shape=(3, 1))
        dec = np.pi / 4 + 2. * np.pi / 180. * tf.random_normal(shape=(3, 1))
        Xd = tf.concat([ra, dec], axis=1)
        Xa = tf.constant(np.concatenate([lofar_array[1][0:1,:], lofar_array[1]],axis=0), dtype=float_type)
        coord_feed = CoordinateFeed(time_feed, Xd, Xa,
                                    coord_map=ITRSToENUWithReferences(lofar_array[1][0,:], [np.pi/4,np.pi/4], lofar_array[1][0,:]))
        init, next = init_feed(coord_feed)
        kern = DTECIsotropicTimeGeneralODE(variance=5e-2,timescale=30.,lengthscales=18., a=300., b=90.,
                                           fed_kernel='RBF',obs_type='DTEC',squeeze=True,
                                           ode_type='fixed',kernel_params={'rtol':1e-6,'atol':1e-6})
        K, info = kern.K(next, full_output=True)
        tf_session.run([init])
        [K] = tf_session.run([K])
        print(info)
        import pylab as plt
        plt.imshow(K)
        plt.colorbar()
        plt.show()


        # L = np.linalg.cholesky(K1 + 1e-6*np.eye(K1.shape[-1]))
        # ddtecs = np.einsum("ab,bc->ac",L, np.random.normal(size=L.shape)).reshape(list(dims)+[L.shape[0]])
        # print(ddtecs[:,:,51].var(), 0.01**2/ddtecs[:,:,51].var())


def test_kernel_equivalence(tf_session, lofar_array):
    with tf_session.graph.as_default():
        index_feed = IndexFeed(1)
        obstime_init = at.Time("2018-01-01T00:00:00.000", format='isot')
        times = obstime_init.mjd*86400. + tf.cast(tf.linspace(0., 50., 9)[:, None],float_type)
        time_feed = TimeFeed(index_feed, times)
        ra = np.pi / 4 + 2. * np.pi / 180. * tf.random_normal(shape=(3, 1))
        dec = np.pi / 4 + 2. * np.pi / 180. * tf.random_normal(shape=(3, 1))
        Xd = tf.concat([ra, dec], axis=1)
        Xa = tf.constant(np.concatenate([lofar_array[1][0:1,:], lofar_array[1]],axis=0), dtype=float_type)
        coord_feed = CoordinateFeed(time_feed, Xd, Xa,
                                    coord_map=tf_coord_transform(itrs_to_enu_with_references(lofar_array[1][0,:], [np.pi/4,np.pi/4], lofar_array[1][0,:])))
        init, next = init_feed(coord_feed)
        kernels = [DTECIsotropicTimeGeneralODE(variance=2e11**2, timescale=30., lengthscales=10., a=300., b=90.,
                   fed_kernel='RBF', obs_type='DDTEC', squeeze=True,
                   ode_type='adaptive', kernel_params={'rtol':1e-6, 'atol':1e-6}),
                   DTECIsotropicTimeGeneralODE(variance=1e9**2,timescale=30.,lengthscales=10., a=300., b=90.,
                                           fed_kernel='RBF',obs_type='DDTEC',squeeze=True,
                                           ode_type='fixed',kernel_params={'resolution':5}),
                   DTECIsotropicTimeGeneral(variance=1e9**2, timescale=30., lengthscales=10., a=300., b=90.,
                                            fed_kernel='RBF', obs_type='DDTEC', squeeze=True,
                                            kernel_params={'resolution': 4}),
                   DTECIsotropicTimeGeneral(variance=1e9**2, timescale=30., lengthscales=10., a=300., b=90.,
                                            fed_kernel='RBF', obs_type='DDTEC', squeeze=True,
                                            kernel_params={'resolution': 6}),
                   DTECIsotropicTimeGeneral(variance=1e9**2, timescale=30., lengthscales=10., a=300., b=90.,
                                            fed_kernel='RBF', obs_type='DDTEC', squeeze=True,
                                            kernel_params={'resolution': 8}),
                   DTECIsotropicTimeGeneral(variance=1e9**2, timescale=30., lengthscales=10., a=300., b=90.,
                                            fed_kernel='RBF', obs_type='DDTEC', squeeze=True,
                                            kernel_params={'resolution': 10}),
                   DTECIsotropicTimeGeneral(variance=1e9**2, timescale=30., lengthscales=10., a=300., b=90.,
                                            fed_kernel='RBF', obs_type='DDTEC', squeeze=True,
                                            kernel_params={'resolution': 12}),
                   DTECIsotropicTimeGeneral(variance=1e9**2, timescale=30., lengthscales=10., a=300., b=90.,
                                            fed_kernel='RBF', obs_type='DDTEC', squeeze=True,
                                            kernel_params={'resolution': 14}),
                   DTECIsotropicTimeGeneral(variance=1e9**2, timescale=30., lengthscales=10., a=300., b=90.,
                                            fed_kernel='RBF', obs_type='DDTEC', squeeze=True,
                                            kernel_params={'resolution': 16}),
                   DTECIsotropicTimeGeneral(variance=1e9**2, timescale=30., lengthscales=10., a=300., b=90.,
                                            fed_kernel='RBF', obs_type='DDTEC', squeeze=True,
                                            kernel_params={'resolution': 18})
                   ]
        K = [kern.K(next) for kern in kernels]
        tf_session.run([init])
        K = tf_session.run(K)
        print(np.sqrt(np.mean(np.diag(K[0]))))
        #1/m^3 km
        for k in K:
            print(np.mean(np.abs(K[1] - k)/np.mean(np.diag(K[1]))))
        # plt.imshow(K[0])
        # plt.colorbar()
        # plt.show()
        # plt.imshow(K[1])
        # plt.colorbar()
        # plt.show()
        # plt.imshow(K[2])
        # plt.colorbar()
        # plt.show()


def test_ddtec_screen(tf_session, lofar_array):
    with tf_session.graph.as_default():
        index_feed = IndexFeed(1)
        obstime_init = at.Time("2018-01-01T00:00:00.000", format='isot')
        times = obstime_init.mjd*86400. + tf.cast(tf.linspace(0., 50., 9)[:, None],float_type)
        time_feed = TimeFeed(index_feed, times)
        M = 20
        ra_vec = np.linspace(np.pi / 4 - 2.*np.pi/180., np.pi / 4 + 2.*np.pi/180., M)
        dec_vec = np.linspace(np.pi / 4 - 2. * np.pi / 180., np.pi / 4 + 2. * np.pi / 180., M)
        ra, dec = np.meshgrid(ra_vec, dec_vec, indexing='ij')
        ra = ra.flatten()[:,None]
        dec = dec.flatten()[:,None]
        Xd = tf.concat([ra, dec], axis=1)
        Xa = tf.constant(np.concatenate([ lofar_array[1][50:51,:]],axis=0), dtype=float_type)
        coord_feed = CoordinateFeed(time_feed, Xd, Xa,
                                    coord_map=tf_coord_transform(itrs_to_enu_with_references(lofar_array[1][0,:], [np.pi/4,np.pi/4], lofar_array[1][0,:])))
        init, next = init_feed(coord_feed)
        kern = DTECIsotropicTimeGeneral(variance=0.1,timescale=30.,lengthscales=15., a=300., b=100.,
                                           fed_kernel='RBF',obs_type='DDTEC',squeeze=True,
                                           kernel_params={'resolution':5})
        K = kern.K(next)
        tf_session.run([init])
        K = tf_session.run(K)

        s = np.mean(np.diag(K))
        K /= s

        L = np.sqrt(s)*np.linalg.cholesky(K + 1e-6*np.eye(K.shape[-1]))
        ddtecs = np.einsum("ab,b->a",L, np.random.normal(size=L.shape[0])).reshape((M,M))
        import pylab as plt
        plt.imshow(ddtecs)
        plt.colorbar()
        plt.savefig("sim_ddtecs.png")
        plt.show()
        plt.imshow(np.sin(-8.448e6 / 140e6 * ddtecs))
        plt.colorbar()
        plt.savefig("sim_ddtecs.png")
        plt.show()