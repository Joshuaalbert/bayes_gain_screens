import tensorflow as tf
from gpflow.kernels import Kernel
from gpflow.params import Parameter
from gpflow import params_as_tensors
from gpflow import settings
from gpflow import transforms
import numpy as np
from .model import HGPR
from .directional_models import gpflow_kernel
float_type = settings.float_type

class NonIntegralKernel(Kernel):
    def __init__(self,
                 ref_direction=[0., 0., 1.],
                 ref_location = [0., 0., 0.],
                 ant_anisotropic=False,
                 dir_anisotropic=False,
                 active_dims=None,
                 dir_kernel = None,
                 ant_kernel=None,
                 obs_type='DDTEC'):
        super().__init__(6, active_dims,
                         name="NonIntKernel_{}{}_{}{}".format('aniso' if dir_anisotropic else "iso",
                                                              dir_kernel.name,
                                                              'aniso' if ant_anisotropic else "iso",
                                                              ant_kernel.name))

        self.dir_kernel = dir_kernel
        self.ant_kernel = ant_kernel

        self.obs_type = obs_type
        self.ref_direction = Parameter(ref_direction,
                                       dtype=float_type, trainable=False)
        self.ref_location = Parameter(ref_location,
                                       dtype=float_type, trainable=False)
        self.ant_anisotropic = ant_anisotropic
        self.dir_anisotropic = dir_anisotropic
        if self.ant_anisotropic:
            # Na, 3, 3
            self.ant_M = Parameter(np.eye(3), dtype=float_type,
                               transform=transforms.LowerTriangular(3, squeeze=True))

        if self.dir_anisotropic:
            # Na, 3, 3
            self.dir_M = Parameter(np.eye(3), dtype=float_type,
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


        k1 = X1[..., 0:3]
        if sym:
            k2 = X2[..., 0:3]
        else:
            k2 = X2[..., 0:3]

        x1 = X1[..., 3:6]
        if sym:
            x2 = X2[..., 3:6]
        else:
            x2 = X2[..., 3:6]

        if self.dir_anisotropic:
            # M_ij.k_nj
            k1 = tf.matmul(k1, self.dir_M, transpose_b=True)
            if sym:
                k2 = k1
            else:
                k2 = tf.matmul(k2, self.dir_M, transpose_b=True)

        if self.ant_anisotropic:
            # M_ij.x_nj
            x1 = tf.matmul(x1, self.ant_M, transpose_b=True)
            if sym:
                x2 = x1
            else:
                x2 = tf.matmul(x2, self.ant_M, transpose_b=True)

        kern_dir = self.dir_kernel
        kern_ant = self.ant_kernel

        Ka00 = kern_ant.K(x1, x2)
        Ka01 = kern_ant.K(x1, self.ref_location[None, :])
        if sym:
            Ka10 = tf.transpose(Ka01)
        else:
            Ka10 = kern_ant.K(self.ref_location[None, :], x2)
        Ka11 = kern_ant.K(self.ref_location[None, :], self.ref_location[None, :])


        Kd00 = kern_dir.K(k1, k2)
        Kd01 = kern_dir.K(k1, self.ref_direction[None, :])
        if sym:
            Kd10 = tf.transpose(Kd01)
        else:
            Kd10 = kern_dir.K(self.ref_direction[None, :], k2)
        Kd11 = kern_dir.K(self.ref_direction[None, :], self.ref_direction[None, :])

        if self.obs_type == 'TEC':
            return Ka00*Kd00
        if self.obs_type == 'DTEC':
            return Kd00 + Kd01 + Kd10 + Kd11
        if self.obs_type == 'DDTEC':
            return (Ka00 + Ka01 + Ka10 + Ka11)*(Kd00 + Kd01 + Kd10 + Kd11)


def generate_models(X, Y, Y_var, ref_direction, ref_location, reg_param = 1., parallel_iterations=10):
    dir_settings = [('RBF', dict(variance=10**2, lengthscales=0.01)),
                    ('M52', dict(variance=10**2, lengthscales=0.01)),
                    ('M32', dict(variance=10**2, lengthscales=0.01)),
                    ('ArcCosine', dict(variance=10**2))]
    ant_settings = [('RBF', dict(variance=1., lengthscales=10.)),
                    ('M52', dict(variance=1., lengthscales=10.)),
                    ('M32', dict(variance=1., lengthscales=10.)),
                    ('ArcCosine', dict(variance=1.))]
    kernels = []
    for a in ant_settings:
        for d in dir_settings:
            kernels.append(NonIntegralKernel(ref_direction=ref_direction,
                                 ref_location=ref_location,
                                ant_anisotropic=False,
                                 dir_anisotropic=False,
                                dir_kernel=gpflow_kernel(d[0], **d[1]),
                                 ant_kernel = gpflow_kernel(a[0], **a[1]),
                                obs_type='DDTEC'))

    models = [HGPR(X, Y, Y_var, kern, regularisation_param=reg_param, parallel_iterations=parallel_iterations,
                   name='HGPR_{}'.format(kern.name))
              for kern in kernels]

    return models