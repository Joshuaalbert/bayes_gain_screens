import tensorflow as tf
import tensorflow_probability as tfp
from .. import float_type
import numpy as np

class Update(object):
    """
    Class that performs conditioning of a prior on data, and observational/Levy covariances.
    """
    def __init__(self, *args, S=200, force_diag_Sigma=False, force_diag_Omega=False, **kwargs):
        self.force_diag_Sigma = force_diag_Sigma
        self.force_diag_Omega = force_diag_Omega
        self.S = S

    def __call__(self, t, prior_mu, prior_Gamma, y, Sigma, *serve_values):
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

    def _forward(self, samples):
        """
        Computes the data-domain samples by pushing forward.
        :param samples: tf.Tensor
            [S, B, K]
        :return: tf.Tensor
            [S, B, N]
        """
        raise NotImplementedError()

    def predictive_distribution(self, post_mu_b, post_Gamma_b):
        # S, B, K
        samples = tfp.distributions.MultivariateNormalFullCovariance(loc=post_mu_b,
                                                                     covariance_matrix=post_Gamma_b).sample(self.S)
        # S, B, N
        y_pred = self._forward(samples)

        post_mean = tf.reduce_mean(y_pred, axis=0)
        post_cov = tfp.stats.covariance(y_pred,sample_axis=0, event_axis=-1)
        return post_mean, post_cov


    def get_params(self, y, post_mu_b, post_Gamma_b):
        """
        If p(X | y, Sigma) = N[post_mu_b, post_Gamma_b]
        then this returns an estimate of the observational covariance, Sigma, and the Levy step covariance.
        Assumes the transfer function is identity (zero-drift).
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
        # Omega_new = tfp.stats.covariance(samples[:, 1:, :] - samples[:, :-1, :], sample_axis=[0, 1], event_axis=2)
        d_samples = samples[:, 1:, :] - samples[:, :-1, :]
        # [K, K]
        Omega_new = tf.reduce_mean(d_samples[:, :, :, None]*d_samples[:, :, None, :], axis=[0,1])
        # S, B, N
        y_pred = self._forward(samples)

        residuals = y - y_pred
        Sigma_new = tf.reduce_mean(residuals[:, :, :, None]*residuals[:, :, None, :], axis=[0,1])
        # Sigma_new = tfp.stats.covariance(y - y_pred, sample_axis=[0, 1], event_axis=2)

        if self.force_diag_Sigma:
            Sigma_new = tf.linalg.diag(tf.linalg.diag_part(Sigma_new))
        if self.force_diag_Omega:
            Omega_new = tf.linalg.diag(tf.linalg.diag_part(Omega_new))

        return Sigma_new, Omega_new

class UpdatePy(Update):
    def __init__(self, *args, **kwargs):
        super(UpdatePy, self).__init__(*args, **kwargs)

    def _update_function(self, t, prior_mu, prior_Gamma, y, Sigma, *serve_values):
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

    def __call__(self, t, prior_mu, prior_Gamma, y, Sigma, *serve_values):
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

        def _call(t, prior_mu, prior_Gamma, y, Sigma, *serve_values):
            serve_values = [v.numpy() for v in serve_values]
            prior_mu, prior_Gamma, y, Sigma = prior_mu.numpy(), prior_Gamma.numpy(), y.numpy(), Sigma.numpy()
            post_mu, post_Gamma = self._update_function(t, prior_mu, prior_Gamma, y, Sigma, *serve_values)

            return [post_mu.astype(np.float64), post_Gamma.astype(np.float64)]

        return tf.py_function(_call, [t, prior_mu, prior_Gamma, y, Sigma]+list(serve_values), [float_type, float_type], name='update')