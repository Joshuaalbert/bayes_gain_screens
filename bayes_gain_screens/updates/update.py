import tensorflow as tf
import tensorflow_probability as tfp
from .. import float_type
import numpy as np

class Update(object):
    """
    Class that performs conditioning of a prior on data, and observational/Levy covariances.
    """
    def __init__(self, *args, S=200, **kwargs):
        self.S = S

    def __call__(self, t, prior_mu, prior_Gamma, y, Sigma):
        """
        If p(X) = N[prior_mu, prior_Gamma]
        then this computes,

        p(X | y, Sigma) = p(y | X, Sigma) p(X) / p(y | Sigma)

        :param prior_mu: tf.Tensor
            [K]
        :param prior_Gamma: tf.Tensor
            [K,K]
        :param y: tf.Tensor
            [N]
        :param Sigma: tf.Tensor
            [N,N]
        :return:
            if p(X | y, Sigma) is a Gaussian then this returns:
            the mean [K] and the covariance [K, K] of type tf.Tensor
        """

        raise NotImplementedError()

    def get_params(self, y, post_mu_b, post_Gamma_b):
        """
        If p(X | y, Sigma) = N[post_mu_b, post_Gamma_b]
        then this returns an estimate of the observational covariance, Sigma, and the Levy step covariance.
        :param y: tf.Tensor
            [B, N]
        :param post_mu_b: tf.Tensor
            [B, K]
        :param post_Gamma_b: tf.Tensor
            [B, K, K]
        :return: the observational covariance and Levy covariance estimates
            [N, N], [K, K] of type tf.Tensor
        """
        # S, B, K
        samples = tfp.distributions.MultivariateNormalFullCovariance(loc=post_mu_b,
                                                                     covariance_matrix=post_Gamma_b).sample(self.S)
        Q_new = tfp.stats.covariance(samples[:, 1:, :] - samples[:, :-1, :], sample_axis=[0, 1], event_axis=2)

        tec_conv = tf.constant(-8.4479745e6 / self.freqs, float_type)

        # S, B, Nf
        phase = samples[:, :, 0:1] * tec_conv
        R_new = tfp.stats.covariance(y - tf.concat([tf.math.cos(phase), tf.math.sin(phase)], axis=-1),
                                     sample_axis=[0, 1], event_axis=2)
        return R_new, Q_new

class UpdatePy(Update):
    def __init__(self, *args, **kwargs):
        super(UpdatePy, self).__init__(*args, **kwargs)

    def _update_function(self, t, prior_mu, prior_Gamma, y, Sigma):
        """
        If p(X) = N[prior_mu, prior_Gamma]
        then this computes,

        p(X | y, Sigma) = p(y | X, Sigma) p(X) / p(y | Sigma)

        :param prior_mu: np.array
            [K]
        :param prior_Gamma: np.array
            [K,K]
        :param y: np.array
            [N]
        :param Sigma: np.array
            [N,N]
        :return:
            if p(X | y, Sigma) is a Gaussian then this returns:
            the mean [K] and the covariance [K, K] of type np.array
        """
        raise NotImplementedError()

    def __call__(self, t, prior_mu, prior_Gamma, y, Sigma):
        """
        If p(X) = N[prior_mu, prior_Gamma]
        then this computes,

        p(X | y, Sigma) = p(y | X, Sigma) p(X) / p(y | Sigma)

        :param prior_mu: tf.Tensor
            [K]
        :param prior_Gamma: tf.Tensor
            [K,K]
        :param y: tf.Tensor
            [N]
        :param Sigma: tf.Tensor
            [N,N]
        :return:
            if p(X | y, Sigma) is a Gaussian then this returns:
            the mean [K] and the covariance [K, K] of type tf.Tensor
        """

        def _call(t, prior_mu, prior_Gamma, y, Sigma):
            prior_mu, prior_Gamma, y, Sigma = prior_mu.numpy(), prior_Gamma.numpy(), y.numpy(), Sigma.numpy()
            post_mu, post_Gamma = self._update_function(t, prior_mu, prior_Gamma, y, Sigma)

            return [post_mu.astype(np.float64), post_Gamma.astype(np.float64)]

        return tf.py_function(_call, [t, prior_mu, prior_Gamma, y, Sigma], [float_type, float_type], name='update')