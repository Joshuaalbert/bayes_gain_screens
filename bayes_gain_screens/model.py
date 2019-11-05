from gpflow.models import GPModel
from gpflow.likelihoods import Gaussian
from gpflow.params import Parameter
from gpflow import DataHolder, name_scope, params_as_tensors, autoflow
from gpflow import settings
float_type = settings.float_type
import tensorflow as tf
import numpy as np
from typing import List
from scipy.special import logsumexp
from gpflow.training import ScipyOptimizer
from . import logging



class HGPR(GPModel):

    def __init__(self, X, Y, Y_var, kern, regularisation_param=1., mean_function=None, parallel_iterations=1,
                 name=None):

        likelihood = Gaussian()
        # M, D
        X = DataHolder(X)
        # B, (T), M
        Y = DataHolder(Y)
        num_latent = Y.shape[0]
        GPModel.__init__(self, X=X, Y=Y, kern=kern, likelihood=likelihood,
                         mean_function=mean_function, num_latent=num_latent, name=name)
        self.Y_var = DataHolder(Y_var)
        self.parallel_iterations = parallel_iterations
        self.regularisation_param = Parameter(regularisation_param, dtype=float_type, trainable=False)

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

class AverageModel(object):

    INF_UNCERT = np.finfo(np.float64).max#np.inf works too, but gradient of lml becomes nan.

    def __init__(self, models: List[HGPR], debug=False, flag_outliers=True):
        self.models = models
        self.debug = debug

    @property
    def models(self):
        return self._models

    @models.setter
    def models(self, value):
        if not isinstance(value, (tuple, list)):
            value = [value]

        for m in value:
            if m.num_latent is not value[0].num_latent:
                raise ValueError("num_latent must be the same")
        self._models = value

    def set_hyperparams(self, hyperparams:dict):
        for m in self.models:
            m.assign(hyperparams[m.name])

    def get_hyperparams(self):
        return {m.name : m.read_trainables() for m in self.models}

    def optimise_and_flag(self):
        Y_var = self.models[0].Y_var.value
        X = self.models[0].X.value
        Y = self.models[0].Y.value
        found_outliers = np.zeros(Y_var.shape, dtype=np.bool)
        for i in range(2):
            g_lml = np.mean([model.grad_likelihood_new_data(X, Y, Y_var) for model in self.models], axis=0)
            g_lml_ = g_lml[~found_outliers].flatten()
            thresh = np.min(g_lml_) + 1.5*np.mean(np.abs(g_lml_ - np.min(g_lml_)))
            found_outliers = g_lml >= thresh
            for model in self.models:
                model.Y_var = np.where(found_outliers, np.inf, Y_var)
                try:
                    ScipyOptimizer().minimize(model)
                except:
                    logging.error("Problem with optimisation!!")
                    model.initialzie(force=True)
                    ScipyOptimizer().minimize(model)
                    pass
        outliers = self.models[0].Y_var.value == np.inf

        logging.info("Flagged {} outliers".format(outliers.sum()))
        with np.printoptions(precision=2):
            logging.info("Learned model:\n{}".format(
                "\n".join(["\t{} -> {}".format(k, v) for (k, v) in model.read_trainables().items()])))

    def optimise(self):
        for model in self.models:
            logging.info("Optimising model: {}".format(model.name))
            opt = ScipyOptimizer()
            try:
                opt.minimize(model)
            except:
                print((model.read_trainables()))
                print(model.X.value)
                print(model.Y.value)
                raise ValueError("Failure")
            with np.printoptions(precision=2):
                logging.info("Learned model:\n{}".format(
                    "\n".join(["\t{} -> {}".format(k, v) for (k,v) in model.read_trainables().items()])))


    def predict_f(self, X, only_mean=True):
        """
        Predict over all batches
        :param X:
        :return:
        """
        post_means = []
        post_vars = []
        log_marginal_likelihoods = []
        for model in self.models:
            if only_mean:
                # [B, (T)], [B, (T), N]
                log_marginal_likelihood, post_mean = model.log_marginal_likelihood_and_predict_f_only_mean(X)
            else:
                # [B, (T)], [B, (T), N], [B, (T), N]
                log_marginal_likelihood, post_mean, post_var = model.log_marginal_likelihood_and_predict_f_mean_and_var(X)
                post_vars.append(post_var)
            post_means.append(post_mean)
            log_marginal_likelihoods.append(log_marginal_likelihood)

        # num_models, batch_size, (T)
        log_marginal_likelihoods = np.stack(log_marginal_likelihoods, axis=0)
        with np.printoptions(precision=2):
            logging.info("Model log marginal likelihoods:")
            if len(log_marginal_likelihoods.shape) == 3:
                for m, l in zip(self.models, log_marginal_likelihoods.sum(-1).sum(-1)):
                    logging.info("\t{} -> {:.2f}".format(m.name, l))
            else:
                for m, l in zip(self.models, log_marginal_likelihoods.sum(-1)):
                    logging.info("\t{} -> {:.2f}".format(m.name, l))
        # batch_size
        # num_models, batch_size, N
        post_means = np.stack(post_means, axis = 0)
        # num_models, batch_size, (T)
        weights = np.exp(log_marginal_likelihoods - logsumexp(log_marginal_likelihoods, axis=0))

        # batch_size, (T),  N
        post_mean = np.sum(weights[..., None]*post_means, axis=0)
        if not only_mean:
            # num_models, batch_size, N
            post_vars = np.stack(post_vars, axis=0)
            # batch_size, N
            post_var = np.sum(weights[..., None]*(post_vars + np.square(post_means)), axis=0) - np.square(post_mean)
            return (weights, log_marginal_likelihoods), post_mean, post_var
        return (weights, log_marginal_likelihoods), post_mean

