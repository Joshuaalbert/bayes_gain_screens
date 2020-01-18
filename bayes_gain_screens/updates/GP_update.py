from .update import Update
from ..misc import diagonal_jitter
from gpflow.models import GPModel
from gpflow.likelihoods import Gaussian
from gpflow.params import Parameter
from gpflow import DataHolder, name_scope, params_as_tensors, autoflow
from gpflow import settings
from gpflow.kernels import Kernel
from typing import List

float_type = settings.float_type
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
import numpy as np
from collections import namedtuple


def safe_grad(x, x_ok, func, safe_x, safe_y):
    """
    Provides a safe gradient of unsafe function.

    :param x_ok: tf.Tensor of bool
        Where x is safe
    :param x:
    :param func:
    :param safe_x:
    :param safe_y:
    :return:
    """
    safe_x = tf.where(x_ok, x, safe_x)
    return tf.where(x_ok, func(safe_x), safe_y)


def flagged_log_prob(x, mean, Sigma, A):
    """
    Computes the log probabiliity of x given mean, and covariance of Sigma + A
    where Sigma is diagonal variances and some values can be infinity.

    Equates to the same thing as flagging those points.
    Not gradient friendly unless safe_grad is used.
    :param x: tf.Tensor
        [..., K]
    :param mean:
        [..., K]
    :param Sigma:
        [..., K, K]
    :param A:
        [..., K, K]
    :return: the log probability broadcasted on leading dimensions.
    """
    sigma = tf.math.sqrt(tf.linalg.diag_part(Sigma))
    L_inv = tf.linalg.diag(tf.math.reciprocal(sigma))
    N = tf.shape(A)[-1]
    B = tf.eye(N, dtype=A.dtype) + tf.linalg.matmul(tf.linalg.matmul(L_inv, A), L_inv, transpose_b=True)
    a = tf.linalg.matmul(L_inv, mean[..., None])[..., 0]
    x = tf.linalg.matmul(L_inv, x[..., None])[..., 0]
    dist = tfp.distributions.MultivariateNormalFullCovariance(loc=a, covariance_matrix=B)
    # - np.sum(np.where(np.isinf(sigma), -np.log(np.sqrt(2*np.pi)), np.log(sigma)))
    return dist.log_prob(x) - tf.reduce_sum(
        tf.where(tf.is_inf(sigma), -np.log(np.sqrt(2 * np.pi)) * tf.ones_like(sigma), tf.math.log(sigma)), axis=-1)


class UpdateGaussianProcess(Update):
    """
    Uses variational inference to condition TEC on Gains.
    """

    def __init__(self, K, hypotheses: List[Kernel], B=1, **kwargs):
        super(UpdateGaussianProcess, self).__init__(**kwargs)
        self.B = B
        self.K = K
        # self.mu_0 = np.zeros([Na, Nd_model], dtype=np.float64).flatten()
        # self.Gamma_0 = np.tile(np.eye(Nd_model, dtype=np.float64)[None, :, :], [Na, 1, 1]).flatten()
        self.hypotheses = hypotheses

    def __call__(self, t, prior_mu, prior_Gamma, y, unused_Sigma, *serve_values):
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
        if self.B > 1:
            # B, K
            prior_mu = tf.reshape(prior_mu, [self.B, self.K])
            # B, K, B, K
            prior_Gamma = tf.reshape(prior_Gamma, [self.B, self.K, self.B, self.K])
            # K, K, B, B
            prior_Gamma = tf.transpose(prior_Gamma, (1, 3, 0, 2))
            # K, K, B
            prior_Gamma = tf.linalg.diag_part(prior_Gamma)
            # B, K, K
            prior_Gamma = tf.transpose(prior_Gamma, (2, 0, 1))
        else:
            # B, K
            prior_mu = prior_mu[None, :]
            # B, K, K
            prior_Gamma = prior_Gamma[None, :, :]

        X, X_s, Sigma = serve_values
        N = tf.shape(X)[0]
        K = tf.shape(X_s)[0]
        K = []
        K_s = []
        K_ss = []
        for fed_kernel in self.hypotheses:
            K.append(tf.tile(fed_kernel.K(X)[None, :, :], [self.B, 1, 1]))
            K_s.append(tf.tile(fed_kernel.K(X_s, X)[None, :, :], [self.B, 1, 1]))
            K_ss.append(tf.tile(fed_kernel.K(X_s)[None, :, :], [self.B, 1, 1]))
        H = len(K)
        # H, B, N, N
        K = tf.stack(K, axis=0, name='K')
        # H, B, K, N
        K_s = tf.stack(K_s, axis=0, name="K_sT")
        # H, B, K, K
        K_ss = tf.stack(K_ss, axis=0, name='K_ss')
        # H, B, K, K
        L_ss = tf.linalg.cholesky(K_ss + diagonal_jitter(self.K))
        # H, B,  K, N
        _F = tf.linalg.triangular_solve(L_ss, K_s, adjoint=False, lower=True)
        # H,B,K,N
        F = tf.linalg.triangular_solve(L_ss, _F, adjoint=True, lower=True)
        # H, B,N, N
        K_sTF = tf.linalg.matmul(_F, _F, transpose_a=True)
        # H, B, K, K
        L_prior_Gamma = tf.tile(tf.linalg.cholesky(prior_Gamma + diagonal_jitter(self.K))[None, :, :, :], [H, 1, 1, 1])
        # H, B, K, N
        LF = tf.linalg.triangular_solve(L_prior_Gamma, F, adjoint=False, lower=True)
        # H, B, N, N
        FTGammaF = tf.linalg.matmul(LF, LF, transpose_a=True)
        # H, B, K, N
        GammaF = tf.linalg.triangular_solve(L_prior_Gamma, LF, adjoint=True, lower=True)

        ###
        # marginal likelihood
        # N
        sigma = tf.math.sqrt(tf.linalg.diag_part(Sigma))
        # N,N
        L_inv = tf.linalg.diag(tf.math.reciprocal(sigma))

        # H, B, K
        prior_mu_ext = tf.tile(prior_mu[None, :, :], [H, 1, 1])
        # H, B, N
        marginal_mean = tf.linalg.matmul(F, prior_mu_ext[:, :, :, None], transpose_a=True)[:, :, :, 0]

        # H, B, N, N
        KmK_sTF = K - K_sTF

        loss_res = namedtuple('LossRes', ['f', 'df'])

        def loss(log_variance):
            variance = tf.reshape(tf.math.exp(log_variance), [H, self.B])
            # H,B
            lml = flagged_log_prob(y, marginal_mean, Sigma, variance[:, :, None, None] * KmK_sTF + FTGammaF)
            # 1
            f = -tf.reduce_sum(lml)
            # H: d/dvar_j sum_i f_i(var_i) = d/dvar_j f_j(var_j)
            df = tf.gradients(f, [log_variance])[0]
            return loss_res(f, df)

        # log_variance = tfp.optimizer.linesearch.hager_zhang(loss, np.log(0.1)).left
        # H,B
        log_variance = tfp.optimizer.bfgs_minimize(loss, tf.zeros([H*self.B], dtype=float_type)).position
        # H,B
        variance = tf.reshape(tf.math.exp(log_variance), [H, self.B])
        A = variance[:, :, None, None] * KmK_sTF + FTGammaF
        # H, B
        lml = flagged_log_prob(y, marginal_mean, Sigma, A)

        ###
        # posterior
        # H, B, N, N
        C = tf.eye(N, dtype=float_type) + tf.linalg.matmul(tf.linalg.matmul(L_inv, A), L_inv, transpose_b=True)
        L_C = tf.linalg.cholesky(C + diagonal_jitter(N))
        # H, B, N, K
        J = tf.linalg.matmul(L_inv, GammaF, transpose_b=True)
        # H, B, N, K
        _J = tf.linalg.triangular_solve(L_C, J, lower=True, adjoint=False)
        # H,B,N, 1
        dy = y[:, None] - tf.linalg.matmul(F, prior_mu[:, :, None], transpose_a=True)
        # H, B, N, 1
        dy = tf.linalg.matmul(L_inv, dy)
        # H, B, N, 1
        _dy = tf.linalg.triangular_solve(L_C, dy, lower=True, adjoint=False)
        # H, B, K
        _post_mean = prior_mu + tf.linalg.matmul(_J, _dy, transpose_a=True)[:, :, :, 0]
        # H, B, K, K
        _post_cov = prior_Gamma - tf.linalg.matmul(_J, _J, transpose_a=True)

        ###
        # marginalise over hypotheses
        # H, B
        weights = tf.math.exp(lml - tf.reduce_logsumexp(lml, axis=0, keepdims=True))
        # B, K
        post_mu = tf.reduce_sum(weights[:, :, None] * _post_mean, axis=0)
        # with tf.control_dependencies([tf.print(_post_mean, post_mu)]):
        # B, K, K
        post_Gamma = tf.reduce_sum(weights[:, :, None, None] * (_post_cov
                                                                + _post_mean[:, :, :, None] * _post_mean[:, :, None, :]
                                                                - post_mu[:, :, None] * post_mu[:, None, :]), axis=0)

        if self.B > 1:
            # B*K
            post_mu = tf.reshape(post_mu, [self.B * self.K])
            # K, K, B
            post_Gamma = tf.transpose(post_Gamma, (1, 2, 0))
            # K, K, B, B
            post_Gamma = tf.linalg.diag(post_Gamma)
            # B,K, B,K
            post_Gamma = tf.transpose(post_Gamma, (2, 0, 3, 1))
            # B* K, B* K
            post_Gamma = tf.reshape(post_Gamma, [self.B * self.K, self.B * self.K])
        else:
            # K
            post_mu = post_mu[0, :]
            # K, K
            post_Gamma = post_Gamma[0, :, :]

        return post_mu, post_Gamma

    def predictive_distribution(self, post_mu_b, post_Gamma_b):
        return post_mu_b, post_Gamma_b

    def get_params(self, y, post_mu_b, post_Gamma_b,*serve_values):
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
        Omega_new = tf.reduce_mean(d_samples[:, :, :, None] * d_samples[:, :, None, :], axis=[0, 1])
        # S, B, N
        y_pred = samples[:, :, :tf.shape(y)[-1]]

        residuals = y - y_pred
        Sigma_new = tf.reduce_mean(residuals[:, :, :, None] * residuals[:, :, None, :], axis=[0, 1])
        # Sigma_new = tfp.stats.covariance(y - y_pred, sample_axis=[0, 1], event_axis=2)

        if self.force_diag_Sigma:
            Sigma_new = tf.linalg.diag(tf.linalg.diag_part(Sigma_new))
        if self.force_diag_Omega:
            Omega_new = tf.linalg.diag(tf.linalg.diag_part(Omega_new))

        return Sigma_new, Omega_new


class ConditionalHGPR(GPModel):

    def __init__(self, X, Y, Y_var, kern, prior_mean, prior_cov, mean_function=None, name=None):

        likelihood = Gaussian()
        # M, D
        X = DataHolder(X)
        # B, (T), M
        Y = DataHolder(Y)
        num_latent = Y.shape[0]
        GPModel.__init__(self, X=X, Y=Y, kern=kern, likelihood=likelihood,
                         mean_function=mean_function, num_latent=num_latent, name=name)
        self.Y_var = DataHolder(Y_var)
        self.prior_mean = DataHolder(prior_mean, fix_shape=True, dtype=float_type, name='prior_mean')
        self.prior_cov = DataHolder(prior_cov, fix_shape=True, dtype=float_type, name='prior_cov')

    @name_scope('common')
    @params_as_tensors
    def _build_common(self):

        # (T), M, M
        Kmm = self.kern.K(self.X)
        # B, (T), M
        Y_std = tf.math.sqrt(self.Y_var)

        M = tf.shape(Kmm)[-1]
        # M, M
        eye = tf.linalg.eye(M, dtype=float_type)
        # B, (T), M, M
        K_sigma = Kmm / (Y_std[..., :, None] * Y_std[..., None, :]) + eye
        # B, (T), M, M
        L = tf.linalg.cholesky(K_sigma)
        # B, (T), M
        Y = self.Y / Y_std
        # B, (T), M, 1
        Ly = tf.linalg.triangular_solve(L, Y[..., :, None])

        return L, Ly, Y_std

    @name_scope('batched_likelihood')
    @params_as_tensors
    def _build_batched_likelihood(self, L, Ly, Y_std):

        M = tf.shape(L)[-1]

        # B, (T)
        maha = -0.5 * tf.reduce_sum(tf.math.square(Ly), axis=[-1, -2])

        # B, (T), M
        x_ok = tf.not_equal(Y_std, tf.constant(np.inf, float_type))
        # B, (T), M
        safe_x = tf.where(x_ok, Y_std, tf.ones_like(Y_std))
        # B, (T)
        logdetL = tf.reduce_sum(tf.where(x_ok, tf.math.log(tf.linalg.diag_part(L) * safe_x), tf.zeros_like(Y_std)),
                                axis=-1)

        constant = 0.5 * np.log(np.sqrt(2. * np.pi)) * tf.cast(M, float_type)
        # B, (T)
        log_marginal_likelihood = maha - self.regularisation_param * logdetL - constant
        return log_marginal_likelihood

    @name_scope('likelihood')
    @params_as_tensors
    def _build_likelihood(self):
        r"""
        Construct a tensorflow function to compute the likelihood.
            \log p(Y | theta).
        """
        L, Ly, Y_std = self._build_common()
        # B, (T)
        log_marginal_likelihood = self._build_batched_likelihood(L, Ly, Y_std)
        # with tf.control_dependencies([tf.print(log_marginal_likelihood)]):
        return tf.reduce_sum(log_marginal_likelihood)

    @name_scope('predict')
    @params_as_tensors
    def _build_predict(self, Xnew, full_cov=False):
        """
        Xnew is a data matrix, the points at which we want to predict.
        This method computes
            p(F* | Y)
        where F* are points on the GP at Xnew, Y are noisy observations at X.
        """
        # [B, (T), M, M], [B, (T), M, 1], [B, (T), M]
        L, Ly, Y_std = self._build_common()

        # (T), M, N
        Kmn = self.kern.K(self.X, Xnew)
        # (T), N, N
        Knn = self.kern.K(Xnew) if full_cov else self.kern.Kdiag(Xnew)

        # B, (T), M, N
        A = tf.linalg.triangular_solve(L, Kmn / Y_std[..., :, None])

        # B, (T), N, 1
        post_mean = tf.matmul(A, Ly, transpose_a=True)[..., :, 0]
        if full_cov:
            # B, (T), N, N
            post_cov = Knn - tf.matmul(A, A, transpose_a=True)
        else:
            # sum_k A[k,i]A[k,j]
            # B, (T), N
            post_cov = Knn - tf.reduce_sum(tf.math.square(A), axis=-2)
        return post_mean, post_cov

    @autoflow((float_type, [None, None]))
    def predict_f(self, Xnew):
        """
        Compute the mean and variance of the latent function(s) at the points
        Xnew.
        """
        return self._build_predict(Xnew)

    @autoflow((float_type, [None, None]))
    def predict_y(self, Xnew):
        """
        Compute the mean and variance of held-out data at the points Xnew
        """
        pred_f_mean, pred_f_var = self._build_predict(Xnew)
        return self.likelihood.predict_mean_and_var(pred_f_mean, pred_f_var)

    @autoflow((float_type, [None, None]))
    @params_as_tensors
    def log_marginal_likelihood_and_predict_f_mean_and_cov(self, Xnew):
        """
        Compute the mean and variance of the latent function(s) at the points
        Xnew.
        """
        L, Ly, Y_std = self._build_common()
        log_marginal_likelihood = self._build_batched_likelihood(L, Ly, Y_std)

        # M, N
        Kmn = self.kern.K(self.X, Xnew)
        # N, N
        Knn = self.kern.K(Xnew)

        # B, M, N
        A = tf.linalg.triangular_solve(L, Kmn / Y_std[..., :, None])

        # B, N
        post_mean = tf.matmul(A, Ly, transpose_a=True)[..., :, 0]

        # B, N, N
        post_cov = Knn - tf.matmul(A, A, transpose_a=True)

        return log_marginal_likelihood, post_mean, post_cov

    @autoflow((float_type, [None, None]))
    @params_as_tensors
    def log_marginal_likelihood_and_predict_f_mean_and_var(self, Xnew):
        """
        Compute the mean and variance of the latent function(s) at the points
        Xnew.
        """
        L, Ly, Y_std = self._build_common()
        log_marginal_likelihood = self._build_batched_likelihood(L, Ly, Y_std)

        # M, N
        Kmn = self.kern.K(self.X, Xnew)
        # N, N
        Knn = self.kern.Kdiag(Xnew)

        # B, M, N
        A = tf.linalg.triangular_solve(L, Kmn / Y_std[..., :, None])

        # B, N
        post_mean = tf.matmul(A, Ly, transpose_a=True)[..., :, 0]

        # sum_k A[k,i]A[k,j]
        # B, N
        post_cov = Knn - tf.reduce_sum(tf.math.square(A), axis=-2)

        return log_marginal_likelihood, post_mean, post_cov

    @autoflow((float_type, [None, None]))
    @params_as_tensors
    def log_marginal_likelihood_and_predict_f_only_mean(self, Xnew):
        """
        Compute the mean and variance of the latent function(s) at the points
        Xnew.
        """
        L, Ly, Y_std = self._build_common()
        log_marginal_likelihood = self._build_batched_likelihood(L, Ly, Y_std)

        # M, N
        Kmn = self.kern.K(self.X, Xnew)

        # B, M, N
        A = tf.linalg.triangular_solve(L, Kmn / Y_std[..., :, None])

        # B, N
        post_mean = tf.matmul(A, Ly, transpose_a=True)[..., :, 0]

        return log_marginal_likelihood, post_mean

    @name_scope('grad_likelihood_new_data')
    @params_as_tensors
    @autoflow((float_type, [None, None]), (float_type, [None, None, None]), (float_type, [None, None, None]))
    def grad_likelihood_new_data(self, Xnew, Y_new, Y_var_new):
        # B, (T), N
        Y_std_new = tf.math.sqrt(Y_var_new)
        # [B, (T), N], [B, (T), N, N]
        post_mean, post_cov = self._build_predict(Xnew, full_cov=True)
        M = tf.shape(post_cov)[-1]
        # N, N
        eye = tf.linalg.eye(M, dtype=float_type)
        # B, (T), N, N
        cov = post_cov / (Y_std_new[..., :, None] * Y_std_new[..., None, :]) + eye
        # B, (T), N, N
        L = tf.linalg.cholesky(cov)
        # B, (T), N
        Y_new = (Y_new - post_mean) / Y_std_new
        # B, (T), N, 1
        Ly = tf.linalg.triangular_solve(L, Y_new[..., :, None])
        # B, (T)
        maha = -0.5 * tf.reduce_sum(tf.math.square(Ly), axis=[-2, -1])
        # B, (T), N
        grad_lml = tf.gradients(tf.reduce_sum(maha), [Y_std_new])[0] / tf.cast(tf.size(Y_new), float_type)
        return grad_lml
