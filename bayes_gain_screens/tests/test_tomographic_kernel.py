from ..tomographic_models import DDTECKernel
from .common_setup import *
import tensorflow as tf
from gpflow.kernels import RBF
import numpy as np
from tensorflow.python import debug as tf_debug



def test_grad(tf_session):
    # tf_session = tf_debug.LocalCLIDebugWrapperSession(tf_session)
    with tf_session.graph.as_default():
        kern = DDTECKernel(a=250., b=100., fed_kernel=RBF(3, lengthscales=10.), ref_direction=[0.5,0.5,np.sqrt(1 - 0.5**2 - 0.5**2)], resolution=4)
        X = np.random.normal(size=(10,6))#tf.random.normal((10,6), dtype=tf.float64)
        X = np.array([[0.52,0.5,np.sqrt(1 - 0.52**2 - 0.5**2), 14,1,0],
                      [0.5,0.52,np.sqrt(1 - 0.5**2 - 0.52**2), 15,0,0]], dtype=np.float64)
        X11 = np.array([[0.52,0.5,np.sqrt(1 - 0.52**2 - 0.5**2), 0.,0.,0.],
                      [0.5,0.52,np.sqrt(1 - 0.5**2 - 0.52**2), 0.,0.,0.]], dtype=np.float64)
        ref_dir = np.array([[0.5,0.5,np.sqrt(1 - 0.5**2 - 0.5**2)]])
        ref_loc = np.array([[0., 0., 0.]])
        K1 = kern.compute_K_symm(X)
        # g = kern.K_grad(X)
        kern.a = 250.1
        K2 = kern.compute_K_symm(X)
        kern.a = 250.
        kern.b = 100.1
        K3 = kern.compute_K_symm(X)
        print((K2-K1)*10., (K3-K1)*10.)

if __name__ == '__main__':
    sess = tf.Session(graph=tf.Graph())
    test_grad(sess)