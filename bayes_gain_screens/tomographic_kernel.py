import tensorflow as tf
from gpflow.kernels import Kernel
from gpflow.params import Parameter
from gpflow import params_as_tensors
from gpflow import settings
from gpflow import transforms
from gpflow import autoflow
import numpy as np
from .model import HGPR
from .directional_models import gpflow_kernel
from tensorflow.python.ops.parallel_for import jacobian

float_type = settings.float_type

from . import KERNEL_SCALE
import pytest


class TomographicKernel(Kernel):
    """
        The tomographic kernel is derived from first principles by assuming a GRF over the electron density, from which
        the TEC, DTEC, and DDTEC kernels can be computed as sums of terms of the form,

            I(a,b,c,d) = iint [K(A(a,b) + s1*B(a,b), A(c,d) + s2*B(c,d))](s1,s2) ds1 ds2

                over the unit square [0,1]x[0,1].

        The 2D trapazoidal rule is used
    """

    def __init__(self,
                 a=250.,
                 b=100.,
                 anisotropic=False,
                 active_dims=None,
                 fed_kernel: Kernel = None,
                 ref_location=None,
                 ref_direction=None,
                 ionosphere_type='flat',
                 obs_type='DDTEC',
                 resolution=8):
        super().__init__(6, active_dims,
                         name="IonoKernel{}_{}{}".format(resolution,
                                                         'aniso' if anisotropic else "iso",
                                                         fed_kernel.name))

        self.resolution = resolution
        self.fed_kernel = fed_kernel
        self.obs_type = obs_type.upper()

        self.ref_direction = Parameter([0., 0., 1.] if ref_direction is None else ref_direction,
                                       dtype=float_type, trainable=False)
        self.ref_location = Parameter([0., 0., 0.] if ref_location is None else ref_location,
                                      dtype=float_type, trainable=False)

        if ionosphere_type == 'curved':
            raise ValueError("Curved not implemented yet.")
        self.ionosphere_type = ionosphere_type
        self.a = Parameter(a, dtype=float_type,
                           transform=transforms.Chain(transforms.Logistic(100., 500.), transforms.positiveRescale(a)))
        self.b_frac = Parameter(b / a, dtype=float_type, transform=transforms.Logistic(0., 0.5))

        self.anisotropic = anisotropic
        if self.anisotropic:
            # Na, 3, 3
            self.M = Parameter(np.eye(3), dtype=float_type,
                               transform=transforms.LowerTriangular(3, squeeze=True))

    def randomize_params(self):
        self.a = np.random.uniform(150., 450.)
        self.b_frac = np.random.uniform(0.1, 0.4)
        self.fed_kernel.variance = np.random.uniform(6., 25.)
        self.fed_kernel.lengthscales = np.random.uniform()

    @property
    @params_as_tensors
    def b(self):
        return self.b_frac * self.a

    @params_as_tensors
    def Kdiag(self, X, presliced=False):
        if not presliced:
            X, _ = self._slice(X, None)
        return tf.linalg.diag_part(self.K(X, None))

    @params_as_tensors
    def calculate_ray_endpoints(self, x, k):
        """
        Calculate s- and Ds.

        :param x: tf.Tensor
            [1,3] or [N, 3]
        :param k: tf.Tensor
            [1,3] or [N, 3]
        :return: tuple of tf.Tensor each of shape
            max(x.shape[0],k.shape[0])
        """
        with tf.name_scope('calculate_ray_endpoints', values=[x, k]):
            if self.ionosphere_type == 'flat':
                # Nk
                sec = tf.math.reciprocal(k[:, 2] + tf.constant(1e-6, dtype=k.dtype), name='secphi')
                # Nk
                bsec = sec * self.b
                # max(Nk,Nx)
                sm = sec * (self.a - 0.5 * self.b + self.ref_location[2] - x[:, 2])
                # max(Nk, Nx)
                Ds = tf.broadcast_to(bsec, tf.shape(sm))
                return sm, Ds
        raise NotImplementedError("curved not implemented")

    @autoflow((float_type, [None, None]), (float_type, [None, None]))
    @params_as_tensors
    def K_grad(self, X1, X2):
        return jacobian(self.K(X1, X2), [self.a, self.b_frac])

    def calculate_y(self, x, k, s):
        """

        :param x: tf.Tensor
            [1,3] or [N, 3]
        :param k: tf.Tensor
            [1,3] or [N, 3]
        :param s: tf.Tensor
            [M]
        :return: tf.Tensor
            [M, max(x.shape[0],k.shape[0]), 3]
        """
        # max(Nx, Nk)
        sm, Ds = self.calculate_ray_endpoints(x, k)
        # M, max(Nx,Nk), 3
        y = (x + sm[:,None] * k) + s[:, None, None] * (Ds[:, None] * k)
        # max(Nx,Nk)
        dsdy = Ds
        return y, dsdy

    @params_as_tensors
    def K(self, X1, X2=None, presliced=False):
        """

        :param X1:
            [N, 6]
        :param X2:
            [H, 6]
        :param presliced:
        :return:
        """
        if not presliced:
            X1, X2 = self._slice(X1, X2)

        k1, x1 = X1[:, 0:3], X1[:, 3:6]
        if X2 is None:
            x2, k2 = x1, k1
        else:
            k2, x2 = X2[:, 0:3], X2[:, 3:6]

        # M, res partitions
        s = tf.cast(tf.linspace(0., 1., self.resolution + 1), dtype=float_type)
        ds = tf.constant(1. / self.resolution, dtype=s.dtype)

        K = [self.compute_inner_kernel(x1,k1, x2, k2, s)]
        shape0 = tf.shape(K[0])
        ref_loc = self.ref_location[None, :]
        ref_dir = self.ref_direction[None,:]
        if self.obs_type in ['DTEC', 'DDTEC']:
            K.append(-self.compute_inner_kernel(ref_loc,k1, x2, k2, s, shape=shape0))
            K.append(-self.compute_inner_kernel(x1,k1, ref_loc, k2, s, shape=shape0))
            K.append(self.compute_inner_kernel(ref_loc,k1, ref_loc, k2, s, shape=shape0))
        if self.obs_type in ['DDTEC']:
            K.append(-self.compute_inner_kernel(x1,ref_dir, x2, k2, s, shape=shape0))
            K.append(self.compute_inner_kernel(ref_loc,ref_dir, x2, k2, s, shape=shape0))
            K.append(self.compute_inner_kernel(x1,ref_dir, ref_loc, k2, s, shape=shape0))
            K.append(-self.compute_inner_kernel(ref_loc,ref_dir, ref_loc, k2, s, shape=shape0))

            K.append(-self.compute_inner_kernel(x1, k1, x2, ref_dir, s, shape=shape0))
            K.append(self.compute_inner_kernel(ref_loc, k1, x2, ref_dir, s, shape=shape0))
            K.append(self.compute_inner_kernel(x1, k1, ref_loc, ref_dir, s, shape=shape0))
            K.append(-self.compute_inner_kernel(ref_loc, k1, ref_loc, ref_dir, s, shape=shape0))

            K.append(self.compute_inner_kernel(x1, ref_dir, x2, ref_dir, s, shape=shape0))
            K.append(-self.compute_inner_kernel(ref_loc, ref_dir, x2, ref_dir, s, shape=shape0))
            K.append(-self.compute_inner_kernel(x1, ref_dir, ref_loc, ref_dir, s, shape=shape0))
            K.append(self.compute_inner_kernel(ref_loc, ref_dir, ref_loc, ref_dir, s, shape=shape0))
        #M, N, M, H
        K = tf.math.accumulate_n(K)

        K = [K[0, :, 0, :],
          K[-1, :, 0, :],
          K[0,  :, -1, :],
          K[-1, :, -1, :],
          2. * tf.reduce_sum(K[-1, :, :, :], axis=[1]),
          2. * tf.reduce_sum(K[0, :, :, :], axis=[1]),
          2. * tf.reduce_sum(K[:, :, -1, :], axis=[0]),
          2. * tf.reduce_sum(K[:, :, 0, :], axis=[0]),
          4. * tf.reduce_sum(K[1:-1, :, 1:-1, :], axis=[0, 2])]
        return (KERNEL_SCALE**2 * 0.25 * ds**2) * tf.accumulate_n(K)

    def compute_inner_kernel(self, x1,k1, x2, k2, s, shape=None):
        # [M, N, 3], [N]
        y1, ds1dy = self.calculate_y(x1, k1, s)
        # [M, H, 3], [H]
        y2, ds2dy = self.calculate_y(x2, k2, s)
        shape1 = tf.shape(y1)[:-1]
        shape2 = tf.shape(y2)[:-1]
        y1 = tf.reshape(y1, (-1,3))
        y2 = tf.reshape(y2, (-1,3))
        # [M*N, M*H]
        kern = self.fed_kernel.K(y1, y2)
        # M,N,M,H
        kern = (ds1dy[:, None, None] * ds2dy) * tf.reshape(kern, tf.concat([shape1, shape2], axis=0))
        if shape is not None:
            kern = tf.broadcast_to(kern, shape)
        return kern



