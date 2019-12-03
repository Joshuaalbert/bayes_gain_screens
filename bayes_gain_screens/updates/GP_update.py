from .update import UpdatePy
from gpflow.models import GPModel
from gpflow.likelihoods import Gaussian
from gpflow.params import Parameter
from gpflow import DataHolder, name_scope, params_as_tensors, autoflow
from gpflow import settings
float_type = settings.float_type
import tensorflow as tf
import numpy as np


class UpdateGaussianProcess(UpdatePy):
    """
    Uses variational inference to condition TEC on Gains.
    """

    def __init__(self, Na, Nd_model, S=200):
        super(UpdateGaussianProcess, self).__init__(S=S)
        self.mu_0 = np.zeros([Na, Nd_model], dtype=np.float64).flatten()
        self.Gamma_0 = np.tile(np.eye(Nd_model, dtype=np.float64)[None, :, :], [Na, 1, 1]).flatten()

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
        #B, (T)
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
        #[B, (T), M, M], [B, (T), M, 1], [B, (T), M]
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