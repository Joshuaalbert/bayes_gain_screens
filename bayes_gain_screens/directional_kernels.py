import numpy as np
import tensorflow as tf
from gpflow import transforms, params_as_tensors
from gpflow import settings

float_type = settings.float_type
from gpflow.kernels import Kernel
from gpflow.params import Parameter


class ThinLayerRBF(Kernel):
    def __init__(self, input_dim, variance, hpd, height, active_dims=None, name=None):
        super().__init__(input_dim, active_dims, name=name)
        self.variance = Parameter(variance,
                                  transform=transforms.positiveRescale(variance),
                                  dtype=settings.float_type)

        self.height_hpd_ratio = Parameter(height / hpd,
                                          transform=transforms.positiveRescale(height / hpd),
                                          dtype=settings.float_type)

    @params_as_tensors
    def sep_in_layer(self, k1, k2):
        eps = tf.constant(1e-6, dtype=k1.dtype)
        # N
        secphi1 = tf.math.reciprocal(k1[:, 2] + eps)
        # M
        secphi2 = tf.math.reciprocal(k2[:, 2] + eps)
        # N, M
        costheta = tf.math.reduce_sum(k1[:, None, :] * k2, axis=-1)
        # N, M
        l = (self.height_hpd_ratio * self.scale_factor()) * tf.math.sqrt(
            secphi1[:, None] ** 2 + secphi2 ** 2 - 2. * (secphi1[:, None] * secphi2) * costheta + eps)

        return l

    def scale_factor(self):
        return 1. / np.sqrt(2 * np.log(2.))

    @params_as_tensors
    def Kdiag(self, X, presliced=False):
        return tf.fill(tf.shape(X)[:-1], self.variance)

    @params_as_tensors
    def K(self, X1, X2=None, presliced=False):
        if not presliced:
            X1, X2 = self._slice(X1, X2)
        if X2 is None:
            X2 = X1
        dist = self.sep_in_layer(X1, X2)
        log_res = tf.math.log(self.variance) - 0.5 * tf.math.square(dist)
        return tf.math.exp(log_res)


class GreatCircleRBF(Kernel):
    def __init__(self, input_dim, variance, hpd, active_dims=None, name=None):
        super().__init__(input_dim, active_dims, name=name)
        self.variance = Parameter(variance,
                                  transform=transforms.positiveRescale(variance),
                                  dtype=settings.float_type)

        self.hpd = Parameter(hpd,
                             transform=transforms.positiveRescale(hpd),
                             dtype=settings.float_type)
        levi_civita = np.zeros((3, 3, 3))
        for a1 in range(3):
            for a2 in range(3):
                for a3 in range(3):
                    levi_civita[a1, a2, a3] = np.sign(a2 - a1) * np.sign(a3 - a1) * np.sign(a3 - a2)

        self.levi_civita = Parameter(levi_civita, dtype=settings.float_type, trainable=False)

    def scale_factor(self):
        return 1. / np.sqrt(2 * np.log(2.))

    @params_as_tensors
    def lengthscales(self):
        return self.hpd / self.scale_factor()

    @params_as_tensors
    def greater_circle(self, a, b):
        """
        Greater circle with broadcast
        :param a: [N,3]
        :param b: [M, 3]
        :return: [N, M]
        """
        # aj,ijk -> aik
        A = tf.linalg.tensordot(a, self.levi_civita, axes=[[1], [1]])
        # aik, bk -> aib
        cross = tf.linalg.tensordot(A, b, axes=[[2], [1]])
        # aib -> ab
        cross_mag = tf.linalg.norm(cross, axis=1)
        # ab
        dot_prod = tf.linalg.tensordot(a, b, axes=[[1], [1]])
        return tf.math.atan2(cross_mag, dot_prod)

    @params_as_tensors
    def Kdiag(self, X, presliced=False):
        return tf.fill(tf.shape(X)[:-1], self.variance)

    @params_as_tensors
    def K(self, X1, X2=None, presliced=False):
        if not presliced:
            X1, X2 = self._slice(X1, X2)
        if X2 is None:
            X2 = X1
        dist = self.greater_circle(X1, X2) / self.lengthscales()
        log_res = tf.math.log(self.variance) - 0.5 * tf.math.square(dist)
        return tf.math.exp(log_res)


class ThinLayerM52(Kernel):
    def __init__(self, input_dim, variance, hpd, height, active_dims=None, name=None):
        super().__init__(input_dim, active_dims, name=name)
        self.variance = Parameter(variance,
                                  transform=transforms.positiveRescale(variance),
                                  dtype=settings.float_type)

        self.height_hpd_ratio = Parameter(height / hpd,
                                          transform=transforms.positiveRescale(height / hpd),
                                          dtype=settings.float_type)

    @params_as_tensors
    def sep_in_layer(self, k1, k2):
        eps = tf.constant(1e-6, dtype=k1.dtype)
        # N
        secphi1 = tf.math.reciprocal(k1[:, 2] + eps)
        # M
        secphi2 = tf.math.reciprocal(k2[:, 2] + eps)
        # N, M
        costheta = tf.math.reduce_sum(k1[:, None, :] * k2, axis=-1)
        # N, M
        l = (self.height_hpd_ratio * self.scale_factor()) * tf.math.sqrt(
            secphi1[:, None] ** 2 + secphi2 ** 2 - 2. * (secphi1[:, None] * secphi2) * costheta + eps)

        return l

    def scale_factor(self):
        return 0.95958

    @params_as_tensors
    def Kdiag(self, X, presliced=False):
        return tf.fill(tf.shape(X)[:-1], self.variance)

    @params_as_tensors
    def K(self, X1, X2=None, presliced=False):
        if not presliced:
            X1, X2 = self._slice(X1, X2)
        if X2 is None:
            X2 = X1
        dist = self.sep_in_layer(X1, X2)
        dist *= np.sqrt(5.)
        dist2 = np.square(dist) / 3.
        log_res = tf.math.log(self.variance) + tf.math.log(1. + dist + dist2) - dist
        return tf.math.exp(log_res)


class GreatCircleM52(Kernel):

    def __init__(self, input_dim, variance, hpd, active_dims=None, name=None):
        super().__init__(input_dim, active_dims, name=name)
        self.variance = Parameter(variance,
                                  transform=transforms.positiveRescale(variance),
                                  dtype=settings.float_type)

        self.hpd = Parameter(hpd,
                             transform=transforms.positiveRescale(hpd),
                             dtype=settings.float_type)
        levi_civita = np.zeros((3, 3, 3))
        for a1 in range(3):
            for a2 in range(3):
                for a3 in range(3):
                    levi_civita[a1, a2, a3] = np.sign(a2 - a1) * np.sign(a3 - a1) * np.sign(a3 - a2)

        self.levi_civita = Parameter(levi_civita, dtype=settings.float_type, trainable=False)

    def scale_factor(self):
        return 0.95958

    @params_as_tensors
    def lengthscales(self):
        return self.hpd / self.scale_factor()

    @params_as_tensors
    def greater_circle(self, a, b):
        """
        Greater circle with broadcast
        :param a: [N,3]
        :param b: [M, 3]
        :return: [N, M]
        """
        # aj,ijk -> aik
        A = tf.linalg.tensordot(a, self.levi_civita, axes=[[1], [1]])
        # aik, bk -> aib
        cross = tf.linalg.tensordot(A, b, axes=[[2], [1]])
        # aib -> ab
        cross_mag = tf.linalg.norm(cross, axis=1)
        # ab
        dot_prod = tf.linalg.tensordot(a, b, axes=[[1], [1]])
        return tf.math.atan2(cross_mag, dot_prod)

    @params_as_tensors
    def Kdiag(self, X, presliced=False):
        return tf.fill(tf.shape(X)[:-1], self.variance)

    @params_as_tensors
    def K(self, X1, X2=None, presliced=False):
        """
        The Matern 5/2 kernel. Functions drawn from a GP with this kernel are twice
        differentiable. The kernel equation is
        k(r) =  σ² (1 + √5r + 5/3r²) exp{-√5 r}
        where:
        r  is the Euclidean distance between the input points, scaled by the lengthscale parameter ℓ,
        σ² is the variance parameter.
        """
        if not presliced:
            X1, X2 = self._slice(X1, X2)
        if X2 is None:
            X2 = X1
        dist = self.greater_circle(X1, X2) / self.lengthscales()
        dist *= np.sqrt(5.)
        dist2 = np.square(dist) / 3.
        log_res = tf.math.log(self.variance) + tf.math.log(1. + dist + dist2) - dist
        return tf.math.exp(log_res)


class ThinLayerM32(Kernel):
    def __init__(self, input_dim, variance, hpd, height, active_dims=None, name=None):
        super().__init__(input_dim, active_dims, name=name)
        self.variance = Parameter(variance,
                                  transform=transforms.positiveRescale(variance),
                                  dtype=settings.float_type)

        self.height_hpd_ratio = Parameter(height / hpd,
                                          transform=transforms.positiveRescale(height / hpd),
                                          dtype=settings.float_type)

    @params_as_tensors
    def sep_in_layer(self, k1, k2):
        eps = tf.constant(1e-6, dtype=k1.dtype)
        # N
        secphi1 = tf.math.reciprocal(k1[:, 2] + eps)
        # M
        secphi2 = tf.math.reciprocal(k2[:, 2] + eps)
        # N, M
        costheta = tf.math.reduce_sum(k1[:, None, :] * k2, axis=-1)
        # N, M
        l = (self.height_hpd_ratio * self.scale_factor()) * tf.math.sqrt(
            secphi1[:, None] ** 2 + secphi2 ** 2 - 2. * (secphi1[:, None] * secphi2) * costheta + eps)

        return l

    def scale_factor(self):
        return 1.032

    @params_as_tensors
    def Kdiag(self, X, presliced=False):
        return tf.fill(tf.shape(X)[:-1], self.variance)

    @params_as_tensors
    def K(self, X1, X2=None, presliced=False):
        if not presliced:
            X1, X2 = self._slice(X1, X2)
        if X2 is None:
            X2 = X1
        dist = self.sep_in_layer(X1, X2)
        dist *= np.sqrt(3.)
        log_res = tf.math.log(self.variance) + tf.math.log(1. + dist) - dist
        return tf.math.exp(log_res)


class GreatCircleM32(Kernel):

    def __init__(self, input_dim, variance, hpd, active_dims=None, name=None):
        super().__init__(input_dim, active_dims, name=name)
        self.variance = Parameter(variance,
                                  transform=transforms.positiveRescale(variance),
                                  dtype=settings.float_type)

        self.hpd = Parameter(hpd,
                             transform=transforms.positiveRescale(hpd),
                             dtype=settings.float_type)
        levi_civita = np.zeros((3, 3, 3))
        for a1 in range(3):
            for a2 in range(3):
                for a3 in range(3):
                    levi_civita[a1, a2, a3] = np.sign(a2 - a1) * np.sign(a3 - a1) * np.sign(a3 - a2)

        self.levi_civita = Parameter(levi_civita, dtype=settings.float_type, trainable=False)

    def scale_factor(self):
        return 1.032

    @params_as_tensors
    def lengthscales(self):
        return self.hpd / self.scale_factor()

    @params_as_tensors
    def greater_circle(self, a, b):
        """
        Greater circle with broadcast
        :param a: [N,3]
        :param b: [M, 3]
        :return: [N, M]
        """
        # aj,ijk -> aik
        A = tf.linalg.tensordot(a, self.levi_civita, axes=[[1], [1]])
        # aik, bk -> aib
        cross = tf.linalg.tensordot(A, b, axes=[[2], [1]])
        # aib -> ab
        cross_mag = tf.linalg.norm(cross, axis=1)
        # ab
        dot_prod = tf.linalg.tensordot(a, b, axes=[[1], [1]])
        return tf.math.atan2(cross_mag, dot_prod)

    @params_as_tensors
    def Kdiag(self, X, presliced=False):
        return tf.fill(tf.shape(X)[:-1], self.variance)

    @params_as_tensors
    def K(self, X1, X2=None, presliced=False):
        """
            The Matern 3/2 kernel. Functions drawn from a GP with this kernel are once
            differentiable. The kernel equation is
            k(r) =  σ² (1 + √3r) exp{-√3 r}
            where:
            r  is the Euclidean distance between the input points, scaled by the lengthscale parameter ℓ,
            σ² is the variance parameter.
        """
        if not presliced:
            X1, X2 = self._slice(X1, X2)
        if X2 is None:
            X2 = X1
        dist = self.greater_circle(X1, X2) / self.lengthscales()
        dist *= np.sqrt(3.)
        log_res = tf.math.log(self.variance) + tf.math.log(1. + dist) - dist
        return tf.math.exp(log_res)


class ThinLayerM12(Kernel):
    def __init__(self, input_dim, variance, hpd, height, active_dims=None, name=None):
        super().__init__(input_dim, active_dims, name=name)
        self.variance = Parameter(variance,
                                  transform=transforms.positiveRescale(variance),
                                  dtype=settings.float_type)

        self.height_hpd_ratio = Parameter(height / hpd,
                                          transform=transforms.positiveRescale(height / hpd),
                                          dtype=settings.float_type)

    @params_as_tensors
    def sep_in_layer(self, k1, k2):
        eps = tf.constant(1e-6, dtype=k1.dtype)
        # N
        secphi1 = tf.math.reciprocal(k1[:, 2] + eps)
        # M
        secphi2 = tf.math.reciprocal(k2[:, 2] + eps)
        # N, M
        costheta = tf.math.reduce_sum(k1[:, None, :] * k2, axis=-1)
        # N, M
        l = (self.height_hpd_ratio * self.scale_factor()) * tf.math.sqrt(
            secphi1[:, None] ** 2 + secphi2 ** 2 - 2. * (secphi1[:, None] * secphi2) * costheta + eps)

        return l

    def scale_factor(self):
        return 1. / np.log(2.)

    @params_as_tensors
    def Kdiag(self, X, presliced=False):
        return tf.fill(tf.shape(X)[:-1], self.variance)

    @params_as_tensors
    def K(self, X1, X2=None, presliced=False):
        if not presliced:
            X1, X2 = self._slice(X1, X2)
        if X2 is None:
            X2 = X1
        dist = self.sep_in_layer(X1, X2)
        log_res = tf.math.log(self.variance) - dist
        return tf.math.exp(log_res)


class GreatCircleM12(Kernel):

    def __init__(self, input_dim, variance, hpd, active_dims=None, name=None):
        super().__init__(input_dim, active_dims, name=name)
        self.variance = Parameter(variance,
                                  transform=transforms.positiveRescale(variance),
                                  dtype=settings.float_type)

        self.hpd = Parameter(hpd,
                             transform=transforms.positiveRescale(hpd),
                             dtype=settings.float_type)
        levi_civita = np.zeros((3, 3, 3))
        for a1 in range(3):
            for a2 in range(3):
                for a3 in range(3):
                    levi_civita[a1, a2, a3] = np.sign(a2 - a1) * np.sign(a3 - a1) * np.sign(a3 - a2)

        self.levi_civita = Parameter(levi_civita, dtype=settings.float_type, trainable=False)

    def scale_factor(self):
        return 1. / np.log(2.)

    @params_as_tensors
    def lengthscales(self):
        return self.hpd / self.scale_factor()

    @params_as_tensors
    def greater_circle(self, a, b):
        """
        Greater circle with broadcast
        :param a: [N,3]
        :param b: [M, 3]
        :return: [N, M]
        """
        # aj,ijk -> aik
        A = tf.linalg.tensordot(a, self.levi_civita, axes=[[1], [1]])
        # aik, bk -> aib
        cross = tf.linalg.tensordot(A, b, axes=[[2], [1]])
        # aib -> ab
        cross_mag = tf.linalg.norm(cross, axis=1)
        # ab
        dot_prod = tf.linalg.tensordot(a, b, axes=[[1], [1]])
        return tf.math.atan2(cross_mag, dot_prod)

    @params_as_tensors
    def Kdiag(self, X, presliced=False):
        return tf.fill(tf.shape(X)[:-1], self.variance)

    @params_as_tensors
    def K(self, X1, X2=None, presliced=False):
        """
        The Matern 1/2 kernel. Functions drawn from a GP with this kernel are not
        differentiable anywhere. The kernel equation is
        k(r) = σ² exp{-r}
        where:
        r  is the Euclidean distance between the input points, scaled by the lengthscale parameter ℓ.
        σ² is the variance parameter
        """
        if not presliced:
            X1, X2 = self._slice(X1, X2)
        if X2 is None:
            X2 = X1
        dist = self.greater_circle(X1, X2) / self.lengthscales()
        log_res = tf.math.log(self.variance) - dist
        return tf.math.exp(log_res)


class ThinLayerRQ(Kernel):
    def __init__(self, input_dim, variance, hpd, height, alpha, active_dims=None, name=None):
        super().__init__(input_dim, active_dims, name=name)
        self.variance = Parameter(variance,
                                  transform=transforms.positiveRescale(variance),
                                  dtype=settings.float_type)

        self.alpha = Parameter(alpha,
                             transform=transforms.positiveRescale(alpha),
                             dtype=settings.float_type)

        self.height_hpd_ratio = Parameter(height / hpd,
                                          transform=transforms.positiveRescale(height / hpd),
                                          dtype=settings.float_type)

    @params_as_tensors
    def sep_in_layer(self, k1, k2):
        eps = tf.constant(1e-6, dtype=k1.dtype)
        # N
        secphi1 = tf.math.reciprocal(k1[:, 2] + eps)
        # M
        secphi2 = tf.math.reciprocal(k2[:, 2] + eps)
        # N, M
        costheta = tf.math.reduce_sum(k1[:, None, :] * k2, axis=-1)
        # N, M
        l = (self.height_hpd_ratio * self.scale_factor()) * tf.math.sqrt(
            secphi1[:, None] ** 2 + secphi2 ** 2 - 2. * (secphi1[:, None] * secphi2) * costheta + eps)

        return l

    @params_as_tensors
    def scale_factor(self):
        return tf.math.reciprocal(
            np.sqrt(2.) * tf.math.sqrt(tf.math.pow(np.sqrt(2.), 1. / self.alpha) - 1.) * tf.math.sqrt(self.alpha))

    @params_as_tensors
    def Kdiag(self, X, presliced=False):
        return tf.fill(tf.shape(X)[:-1], self.variance)

    @params_as_tensors
    def K(self, X1, X2=None, presliced=False):
        if not presliced:
            X1, X2 = self._slice(X1, X2)
        if X2 is None:
            X2 = X1
        dist = self.sep_in_layer(X1, X2)
        log_res = tf.math.log(self.variance) - self.alpha * tf.math.log(1. + dist / (2. * self.alpha))
        return tf.math.exp(log_res)


class GreatCircleRQ(Kernel):
    def __init__(self, input_dim, variance, hpd, alpha=10., active_dims=None, name=None):
        super().__init__(input_dim, active_dims, name=name)
        self.variance = Parameter(variance,
                                  transform=transforms.positiveRescale(variance),
                                  dtype=settings.float_type)

        self.hpd = Parameter(hpd,
                             transform=transforms.positiveRescale(hpd),
                             dtype=settings.float_type)
        self.alpha = Parameter(alpha,
                               transform=transforms.positiveRescale(alpha),
                               dtype=settings.float_type)

        levi_civita = np.zeros((3, 3, 3))
        for a1 in range(3):
            for a2 in range(3):
                for a3 in range(3):
                    levi_civita[a1, a2, a3] = np.sign(a2 - a1) * np.sign(a3 - a1) * np.sign(a3 - a2)

        self.levi_civita = Parameter(levi_civita, dtype=settings.float_type, trainable=False)

    @params_as_tensors
    def scale_factor(self):
        return tf.math.reciprocal(
            np.sqrt(2.) * tf.math.sqrt(tf.math.pow(np.sqrt(2.), 1. / self.alpha) - 1.) * tf.math.sqrt(self.alpha))

    @params_as_tensors
    def lengthscales(self):
        return self.hpd / self.scale_factor()

    @params_as_tensors
    def greater_circle(self, a, b):
        """
        Greater circle with broadcast
        :param a: [N,3]
        :param b: [M, 3]
        :return: [N, M]
        """
        # aj,ijk -> aik
        A = tf.linalg.tensordot(a, self.levi_civita, axes=[[1], [1]])
        # aik, bk -> aib
        cross = tf.linalg.tensordot(A, b, axes=[[2], [1]])
        # aib -> ab
        cross_mag = tf.linalg.norm(cross, axis=1)
        # ab
        dot_prod = tf.linalg.tensordot(a, b, axes=[[1], [1]])
        return tf.math.atan2(cross_mag, dot_prod)

    @params_as_tensors
    def Kdiag(self, X, presliced=False):
        return tf.fill(tf.shape(X)[:-1], self.variance)

    @params_as_tensors
    def K(self, X1, X2=None, presliced=False):
        """
        Rational Quadratic kernel. The kernel equation is
        k(r) = σ² (1 + r² / 2α)^(-α)
        where:
        r  is the Euclidean distance between the input points, scaled by the lengthscale parameter ℓ,
        σ² is the variance parameter,
        α  determines relative weighting of small-scale and large-scale fluctuations.
        For α → ∞, the RQ kernel becomes equivalent to the squared exponential.
        """
        if not presliced:
            X1, X2 = self._slice(X1, X2)
        if X2 is None:
            X2 = X1
        dist = tf.math.square(self.greater_circle(X1, X2) / self.lengthscales())
        log_res = tf.math.log(self.variance) - self.alpha * tf.math.log(1. + dist / (2. * self.alpha))
        return tf.math.exp(log_res)


class VectorAmplitudeWrapper(Kernel):
    def __init__(self,
                 amplitude=None,
                 inner_kernel: Kernel = None):
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
                 inner_kernel: Kernel = None,
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
            # M_ij.k_nj -> k_ni
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
                res = kern_dir.K(k1, k2) \
                      - dir_sym \
                      - tf.transpose(dir_sym, (1, 0)) \
                      + kern_dir.K(self.ref_direction[None, :], self.ref_direction[None, :])
            else:
                res = kern_dir.K(k1, k2) \
                      - kern_dir.K(self.ref_direction[None, :], k2) \
                      - kern_dir.K(k1, self.ref_direction[None, :]) \
                      + kern_dir.K(self.ref_direction[None, :], self.ref_direction[None, :])

        if self.amplitude is not None:
            return tf.math.square(self.amplitude)[:, None, None] * res
        return res
