import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
from .. import float_type
import numpy as np
from ..misc import apply_rolling_func_strided

class Update(object):
    """
    Class that performs conditioning of a prior on data, and observational/Levy covariances.
    """
    def __init__(self, *args, S=100, force_diag_Sigma=False, force_diag_Omega=False, Sigma_window=51, Omega_window=51,
                  **kwargs):
        self.force_diag_Sigma = force_diag_Sigma
        self.force_diag_Omega = force_diag_Omega
        self.S = S
        self.Sigma_window = Sigma_window
        self.Omega_window = Omega_window

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

    def _forward(self, samples, *serve_values):
        """
        Computes the data-domain samples by pushing forward.
        :param samples: tf.Tensor
            [S, B, K]
        :return: tf.Tensor
            [S, B, N]
        """
        raise NotImplementedError()

    def predictive_distribution(self, post_mu_b, post_Gamma_b, *serve_values):
        with tf.name_scope('predictive_distribution'):
            # S, B, K
            samples = tfp.distributions.MultivariateNormalFullCovariance(loc=post_mu_b,
                                                                         covariance_matrix=post_Gamma_b).sample(self.S)
            # S, B, N
            y_pred = self._forward(samples, *serve_values)

            post_mean = tf.reduce_mean(y_pred, axis=0)
            post_cov = tfp.stats.covariance(y_pred,sample_axis=0, event_axis=-1)
            return post_mean, post_cov


    def get_params(self, y, post_mu_b, post_Gamma_b, *serve_values):
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
        with tf.name_scope("get_params"):
            # S, B, K
            samples = tfp.distributions.MultivariateNormalFullCovariance(loc=post_mu_b,
                                                                         covariance_matrix=post_Gamma_b).sample(self.S)
            with tf.name_scope("get_omega"):
                # Omega_new = tfp.stats.covariance(samples[:, 1:, :] - samples[:, :-1, :], sample_axis=[0, 1], event_axis=2)
                d_samples = samples[:, 1:, :] - samples[:, :-1, :]
                def rolling_Omega(d_samples):
                    """
                    :param samples: S, B-1, K
                    :return:
                    """
                    #S, K, B-1
                    d_samples = d_samples.numpy().transpose((0,2,1))
                    if self.force_diag_Omega:
                        #S, K, B-1
                        s = d_samples**2
                    else:
                        #S, K, K, B-1
                        s = d_samples[:, :, None, :]*d_samples[:, None, :, :]
                    # K, (K,) B-1
                    s = apply_rolling_func_strided(lambda x: np.mean(x, axis=-1).mean(0), s, self.Omega_window, piecewise_constant=False)
                    #B-1, (K,) K
                    s = s.T
                    #B, (K,) K
                    return np.concatenate([s[0:1, ...], s], axis=0)
                if self.Omega_window > 0:
                    #B, (K,) K
                    Omega_new = tf.py_function(rolling_Omega, [d_samples], [d_samples.dtype],name='Omega_new_rolling')[0]
                else:
                    if self.force_diag_Omega:
                        #K
                        Omega_new = tf.reduce_mean(d_samples**2, axis=[0, 1])
                    else:
                        # [K, K]
                        Omega_new = tf.reduce_mean(d_samples[:, :, :, None]*d_samples[:, :, None, :], axis=[0,1])
                if self.force_diag_Omega:
                    Omega_new = tf.linalg.diag(Omega_new)

            with tf.name_scope("get_sigma"):
                # S, B, N
                y_pred = self._forward(samples, *serve_values)
                residuals = y - y_pred
                def rolling_Sigma(residuals):
                    """
                    :param samples: S, B, N
                    :return:
                    """
                    #S, N, B
                    residuals = residuals.numpy().transpose((0,2,1))
                    if self.force_diag_Sigma:
                        # S, N, B
                        s = residuals**2
                    else:
                        #S, N, N, B
                        s = residuals[:, :, None, :]*residuals[:, None, :, :]
                    #N, (N,) B
                    s = apply_rolling_func_strided(lambda x: np.mean(x, axis=-1).mean(0), s, self.Sigma_window, piecewise_constant=False)
                    #B, (N,) N
                    return s.T
                if self.Sigma_window > 0:
                    #B, (N,) N
                    Sigma_new = tf.py_function(rolling_Sigma, [residuals], [residuals.dtype], name='Sigma_new_rolling')[0]

                else:
                    if self.force_diag_Sigma:
                        # N
                        Sigma_new = tf.reduce_mean(residuals**2, axis=[0, 1])
                    else:
                        #N, N
                        Sigma_new = tf.reduce_mean(residuals[:, :, :, None]*residuals[:, :, None, :], axis=[0,1])
                        # Sigma_new = tfp.stats.covariance(y - y_pred, sample_axis=[0, 1], event_axis=2)

                if self.force_diag_Sigma:
                    Sigma_new = tf.linalg.diag(Sigma_new)

                if self.Sigma_window > 0:
                    #B,N
                    sigma_diag = tf.math.sqrt(tf.linalg.diag_part(Sigma_new))
                    # S,B,N->B,N
                    outliers = tf.reduce_mean(
                        tf.cast(
                            tf.math.abs(residuals) > 2. * sigma_diag,
                            Sigma_new.dtype),
                        axis=0)
                    Sigma_new = Sigma_new + tf.linalg.diag(tf.where(outliers > 0.5, tf.math.square(4.*sigma_diag), tf.zeros_like(sigma_diag)))

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
        with tf.name_scope("update_py"):
            def _call(t, prior_mu, prior_Gamma, y, Sigma, *serve_values):
                serve_values = [v.numpy() for v in serve_values]
                prior_mu, prior_Gamma, y, Sigma = prior_mu.numpy(), prior_Gamma.numpy(), y.numpy(), Sigma.numpy()
                post_mu, post_Gamma = self._update_function(t, prior_mu, prior_Gamma, y, Sigma, *serve_values)

                return [post_mu.astype(np.float64), post_Gamma.astype(np.float64)]

            return tf.py_function(_call, [t, prior_mu, prior_Gamma, y, Sigma]+list(serve_values), [float_type, float_type], name='update')