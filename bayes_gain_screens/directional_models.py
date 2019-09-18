import tensorflow as tf
from gpflow.kernels import Kernel
from gpflow.params import Parameter
from gpflow import params_as_tensors
from gpflow import settings
from gpflow import transforms
from gpflow import defer_build
import numpy as np
from .model import HGPR
from gpflow.kernels import Matern52, Matern32, Matern12, RBF, ArcCosine, Kernel

float_type = settings.float_type


@tf.custom_gradient
def safe_acos_squared(x):
    safe_x = tf.clip_by_value(x, tf.constant(-1., float_type), tf.constant(1., float_type))
    acos = tf.math.acos(safe_x)
    result = tf.math.square(acos)

    def grad(dy):
        g = -2. * acos / tf.math.sqrt(1. - tf.math.square(safe_x))
        g = tf.where(tf.equal(safe_x, tf.constant(1., float_type)), tf.constant(-2., float_type) * tf.ones_like(g), g)
        g = tf.where(tf.equal(safe_x, tf.constant(-1., float_type)), tf.constant(-100, float_type) * tf.ones_like(g), g)
        with tf.control_dependencies([tf.print(tf.reduce_all(tf.is_finite(g)), tf.reduce_all(tf.is_finite(dy)))]):
            return g * dy

    return result, grad


class ArcCosineEQ(Kernel):

    def __init__(
            self,
            input_dim,
            amplitude=None,
            length_scale=None,
            active_dims=None,
            name='ArcCosineEQ'):

        with tf.name_scope(name) as name:
            super().__init__(input_dim, active_dims, name=name)
            self.amplitude = Parameter(amplitude,
                                       transform=transforms.positiveRescale(amplitude),
                                       dtype=settings.float_type)

            self.length_scale = Parameter(length_scale,
                                          transform=transforms.positiveRescale(length_scale),
                                          dtype=settings.float_type)

    @params_as_tensors
    def Kdiag(self, X, presliced=False):
        return tf.fill(tf.shape(X)[:-1], tf.math.square(self.amplitude))

    @params_as_tensors
    def K(self, x1, x2=None, presliced=False):
        """
        :param x1: tf.Tensor
            [..., b, d]
        :param x2: tf.Tensor
            [..., c, d]
        :return: tf.Tensor
            [..., b, c]

        """
        if not presliced:
            x1, x2 = self._slice(x1, x2)
        if x2 is None:
            x2 = x1
        # ..., b, c
        dot = tf.reduce_sum(
            tf.math.multiply(x1[..., :, None, :], x2[..., None, :, :]), axis=-1)
        log_res = safe_acos_squared(dot)
        log_res *= -0.5
        if self.length_scale is not None:
            log_res /= tf.math.square(self.length_scale)
        if self.amplitude is not None:
            log_res += 2. * tf.math.log(self.amplitude)
        return tf.math.exp(log_res)


class Piecewise(Kernel):
    ALLOWED_Q = [0,1,2,3]
    def __init__(
            self,
            input_dim,
            amplitude=1.,
            length_scale=1.,
            q_order = 0,
            active_dims=None,
            name='Piecewise'):
        if q_order not in self.ALLOWED_Q:
            raise ValueError("Q order {} not allowed".format(q_order))
        with tf.name_scope(name) as name:
            super().__init__(input_dim, active_dims, name="Piecewise_{}".format(q_order))
            self.amplitude = Parameter(amplitude,
                                       transform=transforms.positiveRescale(amplitude),
                                       dtype=settings.float_type)

            self.length_scale = Parameter(length_scale,
                                          transform=transforms.positiveRescale(length_scale),
                                          dtype=settings.float_type)
            self.q_order = q_order

    @params_as_tensors
    def Kdiag(self, X, presliced=False):
        return tf.fill(tf.shape(X)[:-1], tf.math.square(self.amplitude))

    @params_as_tensors
    def K(self, x1, x2=None, presliced=False):
        """
        :param x1: tf.Tensor
            [..., b, d]
        :param x2: tf.Tensor
            [..., c, d]
        :return: tf.Tensor
            [..., b, c]

        see RW. Not working yet

        """
        if not presliced:
            x1, x2 = self._slice(x1, x2)
        if x2 is None:
            x2 = x1

        j = float((self.input_dim//2) + self.q_order + 1)
        J = float(j + self.q_order)
        # ..., b, c
        r = tf.math.sqrt(tf.reduce_sum(
            tf.math.squared_difference(x1[..., :, None, :] / self.length_scale,
                                       x2[..., None, :, :] / self.length_scale), axis=-1))
        oneminusr = tf.nn.relu(1. - r)
        if J == 2.:
            res = tf.math.square(oneminusr)
        else:
            res = tf.math.pow(oneminusr, J)
        if self.q_order == 0:
            pass
        if self.q_order == 1:
            res *= (j+1)*r + 1
        if self.q_order == 2:
            res *= (j**2 + 4.*j + 3.)/3.*tf.math.square(r) \
                   + (3.*j + 6.)/3.*r \
                   + 1.
        if self.q_order == 3:
            res *= (j**3 + 9.*j**2 + 23.*j + 15.)/15. * tf.math.pow(r, 3.) \
                   + (6.*j**2 + 36.*j + 45.)/15. * tf.math.square(r) \
                   + (15.*j + 45.)/15.*r \
                   + 1.

        if self.amplitude is not None:
            res *= tf.math.square(self.amplitude)
        return res


def gpflow_kernel(kernel, dims=3, **kwargs):
    kern_map = dict(RBF=RBF, M32=Matern32, M52=Matern52, M12=Matern12,
                    ArcCosine=ArcCosine, ArcCosineEQ=ArcCosineEQ,
                    Piecewise=Piecewise)
    kern = kern_map.get(kernel, None)
    if kern is None:
        raise ValueError("{} not valid kernel".format(kernel))
    return kern(dims, **kwargs)

class VectorAmplitudeWrapper(Kernel):
    def __init__(self,
                 amplitude=None,
                 inner_kernel:Kernel=None):
        super().__init__(inner_kernel.input_dim, inner_kernel.active_dims, name="VecAmp_{}".format(inner_kernel.name))
        self.inner_kernel = inner_kernel

        if amplitude is not None:
            self.amplitude = Parameter(amplitude,
                                       dtype=float_type, transform=transforms.positive)

    @params_as_tensors
    def Kdiag(self, X, presliced=False):
        if not presliced:
            X, _ = self._slice(X, None)
        return tf.linalg.diag_part(self.K(X, None))

    @params_as_tensors
    def K(self, X1, X2=None, presliced=False):

        res = self.inner_kernel.K(X1, X2, presliced)

        if self.amplitude is not None:
            return tf.math.square(self.amplitude)[:, None, None] * res
        return res

class DirectionalKernel(Kernel):
    def __init__(self,
                 ref_direction=[0., 0., 1.],
                 anisotropic=False,
                 active_dims=None,
                 amplitude=None,
                 inner_kernel=None,
                 obs_type='DDTEC'):
        super().__init__(3, active_dims,
                         name="DirectionalKernel_{}{}".format("aniso" if anisotropic else "iso",
                                                               inner_kernel.name))
        self.inner_kernel = inner_kernel

        self.obs_type = obs_type
        self.ref_direction = Parameter(ref_direction,
                                       dtype=float_type, trainable=False)

        if amplitude is not None:
            self.amplitude = Parameter(amplitude,
                                       dtype=float_type, transform=transforms.positive)
        else:
            self.amplitude = None
        self.anisotropic = anisotropic
        if self.anisotropic:
            # Na, 3, 3
            self.M = Parameter(np.eye(3), dtype=float_type,
                               transform=transforms.LowerTriangular(3, squeeze=True))

    @params_as_tensors
    def Kdiag(self, X, presliced=False):
        if not presliced:
            X, _ = self._slice(X, None)
        return tf.linalg.diag_part(self.K(X, None))

    @params_as_tensors
    def K(self, X1, X2=None, presliced=False):

        if not presliced:
            X1, X2 = self._slice(X1, X2)

        sym = False
        if X2 is None:
            X2 = X1
            sym = True

        k1 = X1
        k2 = X2

        if self.anisotropic:
            # M_ij.k_nj
            k1 = tf.matmul(k1, self.M, transpose_b=True)
            if sym:
                k2 = k1
            else:
                k2 = tf.matmul(k2, self.M, transpose_b=True)

        kern_dir = self.inner_kernel
        res = None
        if self.obs_type == 'TEC' or self.obs_type == 'DTEC':
            res = kern_dir.K(k1, k2)
        if self.obs_type == 'DDTEC':
            if sym:
                dir_sym = kern_dir.K(k1, self.ref_direction[None, :])
                res = kern_dir.K(k1, k2) - dir_sym - tf.transpose(dir_sym, (1, 0)) + kern_dir.K(
                    self.ref_direction[None, :], self.ref_direction[None, :])
            res = kern_dir.K(k1, k2) - kern_dir.K(self.ref_direction[None, :], k2) - kern_dir.K(k1,
                                                                                                self.ref_direction[None,
                                                                                                :]) + kern_dir.K(
                self.ref_direction[None, :], self.ref_direction[None, :])

        if self.amplitude is not None:
            return tf.math.square(self.amplitude)[:, None, None]*res
        return res



def generate_models(X, Y, Y_var, ref_direction, reg_param=1., parallel_iterations=10, anisotropic=False):
    amplitude = None
    if len(Y.shape) == 3:
        amplitude = np.ones(Y.shape[1])
    dir_kernels = [  #gpflow_kernel('Piecewise', q_order=0, dims=3, amplitude=10., length_scale=0.01),
                     # gpflow_kernel('Piecewise', q_order=1, dims=3, amplitude=10., length_scale=0.01),
                     # gpflow_kernel('Piecewise', q_order=2, dims=3, amplitude=10., length_scale=0.01),
                     # gpflow_kernel('Piecewise', q_order=3, dims=3, amplitude=10., length_scale=0.01),
        # gpflow_kernel('ArcCosineEQ', dims=3, amplitude=10., length_scale=0.01),
        gpflow_kernel('RBF', dims=3, variance=10. ** 2, lengthscales=0.01),
        gpflow_kernel('M52', dims=3, variance=10. ** 2, lengthscales=0.01),
        gpflow_kernel('M32', dims=3, variance=10. ** 2, lengthscales=0.01),
        gpflow_kernel('M12', dims=3, variance=10. ** 2, lengthscales=0.01),
        gpflow_kernel('ArcCosine', dims=3, variance=10. ** 2)]

    kernels = []
    for d in dir_kernels:
        kernels.append(DirectionalKernel(ref_direction=ref_direction,
                                         anisotropic=anisotropic,
                                         inner_kernel=d,
                                         amplitude=amplitude,
                                         obs_type='DDTEC'))
    dir_kernels = [  # gpflow_kernel('Piecewise', q_order=0, dims=3, amplitude=10., length_scale=0.01),
        # gpflow_kernel('Piecewise', q_order=1, dims=3, amplitude=10., length_scale=0.01),
        # gpflow_kernel('Piecewise', q_order=2, dims=3, amplitude=10., length_scale=0.01),
        # gpflow_kernel('Piecewise', q_order=3, dims=3, amplitude=10., length_scale=0.01),
        # gpflow_kernel('ArcCosineEQ', dims=3, amplitude=10., length_scale=0.01),
        gpflow_kernel('RBF', dims=3, variance=10. ** 2, lengthscales=0.01),
        gpflow_kernel('M52', dims=3, variance=10. ** 2, lengthscales=0.01),
        gpflow_kernel('M32', dims=3, variance=10. ** 2, lengthscales=0.01),
        gpflow_kernel('M12', dims=3, variance=10. ** 2, lengthscales=0.01),
        gpflow_kernel('ArcCosine', dims=3, variance=10. ** 2)]

    for d in dir_kernels:
        kernels.append(VectorAmplitudeWrapper(
                                         inner_kernel=d,
                                         amplitude=amplitude))

    models = [HGPR(X, Y, Y_var, kern, regularisation_param=reg_param, parallel_iterations=parallel_iterations,
                   name="HGPR_{}".format(kern.name))
              for kern in kernels]

    return models

