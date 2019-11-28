import tensorflow as tf
float_type = tf.float64
from bayes_gain_screens.misc import diagonal_jitter
import tensorflow as tf
from scipy.linalg import solve_triangular
from scipy.optimize import minimize, brute
import numpy as np
from scipy.stats import multivariate_normal
import tensorflow_probability as tfp


def constrain(v, a, b):
    return a + (np.tanh(v) + 1) * (b - a) / 2.


def deconstrain(v, a, b):
    return np.arctanh(np.clip((v - a) * 2. / (b - a) - 1., -0.999, 0.999))


def constrain_cor(v):
    return constrain(v, -0.999, 0.999)


def deconstrain_cor(v):
    return deconstrain(v, -0.999, 0.999)


def constrain_tec(v):
    return constrain(v, 0.01, 55.)


def deconstrain_tec(v):
    return deconstrain(v, 0.01, 55.)


def constrain_const(v):
    return constrain(v, 0.001, 2 * np.pi)


def deconstrain_const(v):
    return deconstrain(v, 0.001, 2 * np.pi)


def wrap(p):
    return np.arctan2(np.sin(p), np.cos(p))


def log_gaussian_pdf(x, y, sigma):
    # S, Nf
    x = x / sigma
    y = y / sigma
    log_prob = -0.5 * np.sum(x ** 2 + y ** 2, axis=-1)
    log_prob += -0.5 * sigma.shape[0] * np.log(2 * np.pi) - np.sum(np.log(sigma))
    return log_prob


def log_mv_gaussian_pdf(dx, L_Sigma):
    """
    MV Gaussian (bathces)
    :param dx: [B, Nf]
    :param Sigma: [Nf, Nf]
    """
    # B
    maha = -0.5 * np.sum(solve_triangular(L_Sigma, dx.T, lower=True) ** 2, axis=0)
    log_det = - np.sum(np.log(np.diag(L_Sigma)))
    constant = -0.5 * L_Sigma.shape[0] * np.log(2 * np.pi)
    return maha + log_det + constant

    return multivariate_normal(mean=x, cov=Sigma).logpdf(np.zeros_like(x)) + multivariate_normal(mean=y,
                                                                                                 cov=Sigma).logpdf(
        np.zeros_like(y))


class SolveLossVI(object):
    """
    This class builds the loss function.
    Simple use case:
    # loop over data
    loss_fn = build_loss(Yreal, Yimag, freqs, gain_uncert=0.02, tec_mean_prior=0., tec_uncert_prior=0.1,S=20)
    #brute force
    tec_mean, tec_uncert = brute(loss_fn, (slice(-200, 200,1.), slice(np.log(0.01), np.log(10.), 1.), finish=fmin)
    #The results are Bayesian estimates of tec mean and uncert.

    :param Yreal: np.array shape [Nf]
        The real data (including amplitude)
    :param Yimag: np.array shape [Nf]
        The imag data (including amplitude)
    :param freqs: np.array shape [Nf]
        The freqs in Hz
    :param gain_uncert: float
        The uncertainty of gains.
    :param tec_mean_prior: float
        the prior mean for tec in mTECU
    :param tec_uncert_prior: float
        the prior tec uncert in mTECU
    :param S: int
        Number of hermite terms for Guass-Hermite quadrature
    :return: callable function of the form
        func(params) where params is a tuple or list with:
            params[0] is tec_mean in mTECU
            params[1] is log_tec_uncert in log[mTECU]
        The return of the func is a scalar loss to be minimised.
    """

    def __init__(self, Yreal, Yimag, freqs, tec_mean_prior=0., tec_uncert_prior=100.,
                 const_mean_prior=0., const_uncert_prior=100.,
                 S=20, sigma=None, L_Sigma=None):
        self.x, self.w = np.polynomial.hermite.hermgauss(S)
        self.x_tec = self.x
        self.x_const = self.x
        self.w /= np.pi ** (0.5)
        self.w_tec = self.w
        self.w_const = self.w

        # S_tec
        self.const = const_mean_prior + np.sqrt(2.) * const_uncert_prior * self.x_const

        self.tec_conv = -8.4479745e6 / freqs
        # Nf
        self.Yreal = Yreal
        self.Yimag = Yimag

        self.tec_mean_prior = tec_mean_prior
        self.tec_uncert_prior = tec_uncert_prior
        self.const_mean_prior = const_mean_prior
        self.const_uncert_prior = const_uncert_prior

        self.sigma = sigma
        self.L_Sigma = L_Sigma

    def _scalar_KL(self, mean, uncert, mean_prior, uncert_prior):
        # Get KL
        q_var = np.square(uncert)
        var_prior = np.square(uncert_prior)
        trace = q_var / var_prior
        mahalanobis = np.square(mean - mean_prior) / var_prior
        constant = -1.
        logdet_qcov = np.log(var_prior / q_var)
        twoKL = mahalanobis + constant + logdet_qcov + trace
        prior_KL = 0.5 * twoKL
        return prior_KL

    def test_loss_func(self, params):
        tec_mean, _tec_uncert = params[0], params[1]
        tec_uncert = constrain_tec(_tec_uncert)

        q = multivariate_normal(tec_mean, tec_uncert ** 2)
        tec_prior = multivariate_normal(self.tec_mean_prior, self.tec_uncert_prior ** 2)
        q_samples = q.rvs(1000)
        tec_prior_KL = np.mean(q.logpdf(q_samples) - tec_prior.logpdf(q_samples))
        #         print("tec_prior_KL", tec_prior_KL)

        # S_tec, Nf
        phase = q_samples[:, None] * self.tec_conv
        Yreal_m = np.cos(phase)
        Yimag_m = np.sin(phase)
        # S_tec
        log_prob = log_gaussian_pdf((Yreal_m - self.Yreal),
                                    (Yimag_m - self.Yimag),
                                    self.sigma)
        # scalar
        var_exp = np.mean(log_prob)
        #         print('var_exp',var_exp)
        loss = np.negative(var_exp - tec_prior_KL)
        return loss

    def loss_func(self, params):
        """
        VI loss
        :param params: tf.Tensor
            shapfrom scipy.optimize import brute, fmine [D]
        :return: tf.Tensor
            scalar The loss
        """

        tec_mean, _tec_uncert = params[0], params[1]

        tec_uncert = constrain_tec(_tec_uncert)

        # S_tec
        tec = tec_mean + np.sqrt(2.) * tec_uncert * self.x_tec
        # S_tec, Nf
        phase = tec[:, None] * self.tec_conv
        Yreal_m = np.cos(phase)
        Yimag_m = np.sin(phase)

        dx = np.concatenate([Yreal_m - self.Yreal, Yimag_m - self.Yimag], axis=1)
        log_prob = log_mv_gaussian_pdf(dx, self.L_Sigma)

        # S_tec

        # scalar
        var_exp = np.sum(log_prob * self.w_tec, axis=0)

        # Get KL
        tec_prior_KL = self._scalar_KL(tec_mean, tec_uncert, self.tec_mean_prior, self.tec_uncert_prior)

        loss = np.negative(var_exp - tec_prior_KL)
        if ~np.isfinite(loss):
            raise ValueError(
                "Invalid loss {} var exp {} tec KL {} params {} sigma {}".format(loss, var_exp,
                                                                                 tec_prior_KL,
                                                                                 params,
                                                                                 self.sigma))
        # B
        return loss


def update_step(prior_mu, prior_Gamma, Y, Sigma, freqs, tec_min=-300., tec_max = 300., spacing=10.):
    """
    Perform a single VI optimisation (update step).

    :param prior_mu: [2]
    :param prior_Gamma: [2,2]
    :param Y: [N] complex
    :param Sigma: [N,N]
    :param freqs: [N]
    :return: [2], [2,2]
    """
    Nf = freqs.shape[0]

    try:
        L_Sigma = np.linalg.cholesky(Sigma + 1e-4 * np.eye(2 * Nf))
    except:
        L_Sigma = np.diag(np.sqrt(np.diag(Sigma + 1e-4 * np.eye(2 * Nf))))
    gains = Y[:Nf] + 1j * Y[Nf:]

    s = SolveLossVI(gains.real, gains.imag, freqs,
                    tec_mean_prior=prior_mu[0], tec_uncert_prior=np.sqrt(prior_Gamma[0, 0]),
                    S=20, L_Sigma=L_Sigma)

    sol1 = brute(lambda p: s.loss_func([p[0], deconstrain_tec(5.)]),
                 (slice(tec_min, tec_max, spacing),))

    sol3 = minimize(s.loss_func,
                    np.array([sol1[0], deconstrain_tec(5.)]),
                    method='BFGS').x

    tec_mean = sol3[0]
    tec_uncert = constrain_tec(sol3[1])

    post_mu = np.array([tec_mean, 0], np.float64)
    post_cov = np.array([[tec_uncert ** 2, 0.], [0., 1. ** 2]], np.float64)

    return [post_mu, post_cov]


class Update(object):
    def __init__(self, freqs, tec_scale=300., spacing=10., S=200):
        self.freqs = freqs
        self.S = S
        self.tec_scale = tec_scale
        self.spacing = spacing

    def __call__(self, prior_mu, prior_Gamma, y, Sigma):
        """
        Get the variational posterior.

        :param prior_mu:
            [K]
        :param prior_Gamma:
            [K,K]
        :param y:
            [N]
        :param Sigma:
            [N,N]
        :return:
            [K], [K,K]
        """

        def _call(prior_mu, prior_Gamma, y, Sigma):
            prior_mu, prior_Gamma, y, Sigma = prior_mu.numpy(), prior_Gamma.numpy(), y.numpy(), Sigma.numpy()
            post_mu, post_Gamma = update_step(prior_mu, prior_Gamma, y, Sigma, self.freqs,
                                              -self.tec_scale, self.tec_scale, self.spacing)
            return [post_mu.astype(np.float64), post_Gamma.astype(np.float64)]

        return tf.py_function(_call, [prior_mu, prior_Gamma, y, Sigma], [float_type, float_type], name='updater')

    def get_params(self, y, post_mu_b, post_Gamma_b):
        """
        :param y: [B, N]
        :param post_mu_b: [B, K]
        :param post_Gamma_b: [B, K, K]

        :return: [S, B, N]
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

class NLDSSmoother(object):
    def __init__(self, num_latent, num_observables, sequence_length, update, get_params=None, momentum=0.5):
        """
        Perform non-linear dynamics smoothing.
        """
        N = 2 * num_observables
        B = sequence_length
        K = num_latent
        self._update = update
        self._get_params = get_params

        graph = tf.Graph()

        with graph.as_default():
            Nmax_pl = tf.placeholder(tf.int32, shape=(), name='Nmax')
            # N,N
            Sigma_0_pl = tf.placeholder(float_type, shape=[N, N], name='Sigma_0')
            # K,K
            Omega_0_pl = tf.placeholder(float_type, shape=[K, K], name='Omega_0')
            # B, N
            y_pl = tf.placeholder(float_type, shape=[None, N], name='y')
            B = tf.shape(y_pl)[0]
            mu_0_pl = tf.placeholder(float_type, shape=[K], name='mu_0')
            Gamma_0_pl = tf.placeholder(float_type, shape=[K, K], name='Gamma_0')

            ###
            # Bayesian evidence

            def cond(n, *args):
                return n < Nmax_pl

            def body(n, mu_0_n1, Gamma_0_n1, Sigma_n1, Omega_n1, post_mu_n1, post_Gamma_n1):
                prior_Gamma, post_mu_f, post_Gamma_f = self.forward_filter(y_pl, mu_0_n1, Gamma_0_n1,
                                                                           Sigma_n1, Omega_n1)

                post_mu_b, post_Gamma_b, post_Gamma_inter = self.backward_filter(prior_Gamma, post_mu_f,
                                                                                 post_Gamma_f, Omega_n1)

                res = self.parameter_estimation(y_pl, post_mu_b, post_Gamma_b, post_Gamma_inter)
                Sigma = momentum * Sigma_n1 + (1. - momentum) * res['R_new']
                Omega = momentum * Omega_n1 + (1. - momentum) * res['Q_new']

                mu_0 = mu_0_n1  # momentum*mu_0_n1 + (1. - momentum)*res['mu_0']#mu_0_n1#
                Gamma_0 = Gamma_0_n1  # momentum * Gamma_0_n1 + (1. - momentum) * res['Gamma_0']#Gamma_0_n1#

                post_Gamma_b.set_shape(post_Gamma_n1.shape)
                post_mu_b.set_shape(post_mu_n1.shape)

                Omega.set_shape(Omega_n1.shape)
                Sigma.set_shape(Sigma_n1.shape)

                return [n + 1, mu_0, Gamma_0, Sigma, Omega, post_mu_b, post_Gamma_b]

            [_, mu_0, Gamma_0, Sigma, Omega, post_mu, post_Gamma] = \
                tf.while_loop(cond,
                              body,
                              [tf.constant(0, tf.int32),
                               mu_0_pl,
                               Gamma_0_pl,
                               Sigma_0_pl,
                               Omega_0_pl,
                               tf.zeros([B, K], float_type),
                               tf.zeros([B, K, K], float_type)
                               ])

            #             # [B,K].[N,K]->[B,N]
            #             predictive_y = tf.linalg.matmul(post_mu, A, transpose_b=True)
            #             # B, N, N
            #             predictive_y_Cov = Sigma + tf.linalg.matmul(A_ext, tf.linalg.matmul(post_Gamma, A_ext, transpose_b=True))

            self.y_pl = y_pl
            self.Sigma_0_pl = Sigma_0_pl
            self.Omega_0_pl = Omega_0_pl
            self.mu_0_pl = mu_0_pl
            self.Gamma_0_pl = Gamma_0_pl
            self.Nmax_pl = Nmax_pl
            self.sess = tf.Session(graph=graph)
            self.result = dict(
                post_mu=post_mu,
                post_Gamma=post_Gamma,
                Omega=Omega,
                Sigma=Sigma,
                mu_0=mu_0,
                Gamma_0=Gamma_0)

    #                 predictive_y=predictive_y,
    #                 predictive_y_Cov=predictive_y_Cov)

    def parameter_estimation(self, y, post_mu_b, post_Gamma_b, post_Gamma_inter):
        """
        M-step
        :param y: [B, N]
        :param post_mu: [B, K]
        :param Pt: [B, K, K]
        :param Ptt1: [B, K, K]
        :return:
        """
        B = tf.shape(y)[0]
        N = tf.shape(y)[1]
        K = tf.shape(post_Gamma_b)[2]

        ###
        # observation covariance estimate
        # S, B, N
        R_new, Q_new = self._update.get_params(y, post_mu_b, post_Gamma_b)

        mu_0 = post_mu_b[0, :]
        Gamma_0 = post_Gamma_b[0, :, :]

        return dict(R_new=R_new,
                    Q_new=Q_new,
                    mu_0=mu_0,
                    Gamma_0=Gamma_0)

    def backward_filter(self, prior_Gamma, post_mu_f, post_Gamma_f, Omega):
        """

        http://mlg.eng.cam.ac.uk/zoubin/course04/tr-96-2.pdf

        Perform a forward filter pass
        :param A: tf.Tensor
            [N, K]
        :param prior_Gamma_: tf.Tensor
            [B, K, K]
        :param post_mu_: tf.Tensor
            [B,K]
        :param post_Gamma_: tf.Tensor
            [B,K,K]
        :param Sigma: tf.Tensor
            [N, N]
        :param Omega: tf.Tensor
            [K,K]
        :return: tuple
            post_mu [B,K]
            post_Gamma [B, K, K]
        """

        B = tf.shape(post_Gamma_f)[0]
        K = tf.shape(post_mu_f)[1]

        # JT_aug shifted so that n accesses n-1
        # B,K,K
        # J'_t = (V_t+1^t)^-1 . V_t^t
        JT_aug = tf.concat([tf.zeros([1, K, K], dtype=float_type),
                            tf.linalg.lstsq(prior_Gamma[1:, :, :], post_Gamma_f[:-1, :, :], fast=False)],
                           axis=0)

        def _idx(nm1):
            # B = 3, n1=2 -> 0
            # B = 3, n1=0 -> B-1
            return B - nm1 - 1

        def cond(nm1, *args):
            return nm1 >= 0

        def body(nm1, post_mu_n, post_Gamma_n, post_Gamma_inter_n, post_mu_ta, post_Gamma_ta, post_Gamma_inter_ta):
            #
            n = nm1 + 1
            nm2 = nm1 - 1
            place_nm1 = _idx(nm1)

            prior_Gamma_n = prior_Gamma[n, :, :]
            post_Gamma_f_nm1 = post_Gamma_f[nm1, :, :]
            post_mu_f_nm1 = post_mu_f[nm1, :]

            # K,K
            JT_nm1 = JT_aug[nm1 + 1, :, :]
            JT_nm2 = JT_aug[nm2 + 1, :, :]

            # K
            post_mu_nm1 = post_mu_f_nm1 + \
                          tf.linalg.matmul(JT_nm1, (post_mu_n - post_mu_f_nm1)[:, None],
                                           transpose_a=True)[:, 0]

            # K,K
            post_Gamma_nm1 = post_Gamma_f_nm1 + tf.linalg.matmul(JT_nm1,
                                                                 tf.linalg.matmul(post_Gamma_n - prior_Gamma_n, JT_nm1),
                                                                 transpose_a=True)

            # K,K
            post_Gamma_inter_nm1 = tf.linalg.matmul(
                post_Gamma_f_nm1 + tf.linalg.matmul(JT_nm1, post_Gamma_inter_n - post_Gamma_f_nm1, transpose_a=True),
                JT_nm2)

            return [nm1 - 1, post_mu_nm1, post_Gamma_nm1, post_Gamma_inter_nm1,
                    post_mu_ta.write(place_nm1, post_mu_nm1),
                    post_Gamma_ta.write(place_nm1, post_Gamma_nm1),
                    post_Gamma_inter_ta.write(place_nm1, post_Gamma_inter_nm1)]

        post_mu_ta = tf.TensorArray(float_type, size=B)
        post_Gamma_ta = tf.TensorArray(float_type, size=B)
        post_Gamma_inter_ta = tf.TensorArray(float_type, size=B)

        post_mu_B = post_mu_f[-1, :]
        post_Gamma_B = post_Gamma_f[-1, :, :]
        # K, K
        post_Gamma_inter_B = tf.linalg.lstsq(
            tf.eye(K, dtype=float_type) + tf.linalg.lstsq(post_Gamma_f[-1, :, :], Omega, fast=False) - tf.linalg.lstsq(
                prior_Gamma[-1, :, :], Omega, fast=False),
            post_Gamma_f[-2, :, :], fast=False)

        post_mu_ta = post_mu_ta.write(0, post_mu_B)
        post_Gamma_ta = post_Gamma_ta.write(0, post_Gamma_B)
        post_Gamma_inter_ta = post_Gamma_inter_ta.write(0, post_Gamma_inter_B)

        [_, _, _, _,
         post_mu_ta, post_Gamma_ta, post_Gamma_inter_ta] = \
            tf.while_loop(cond,
                          body,
                          [B - 2,
                           post_mu_B,
                           post_Gamma_B,
                           post_Gamma_inter_B,
                           post_mu_ta,
                           post_Gamma_ta,
                           post_Gamma_inter_ta])

        post_mu = post_mu_ta.stack()[::-1, ...]
        post_Gamma = post_Gamma_ta.stack()[::-1, ...]
        post_Gamma_inter = post_Gamma_inter_ta.stack()[::-1, ...]

        return post_mu, post_Gamma, post_Gamma_inter

    def forward_filter(self, y, mu_0, Gamma_0, Sigma, Omega):
        B = tf.shape(y)[0]

        def cond(n, *args):
            return n < B

        def body(n, post_mu_n1, post_Gamma_n1, prior_Gamma_ta, post_mu_ta, post_Gamma_ta):
            ###
            # get prior
            prior_mu_n = post_mu_n1
            # K,K
            prior_Gamma_n = post_Gamma_n1 + Omega

            post_mu_n, post_Gamma_n = self._update(prior_mu_n, prior_Gamma_n, y[n, :], Sigma)
            post_mu_n.set_shape(post_mu_n1.shape)
            post_Gamma_n.set_shape(post_Gamma_n1.shape)
            return [n + 1, post_mu_n, post_Gamma_n, prior_Gamma_ta.write(n, prior_Gamma_n),
                    post_mu_ta.write(n, post_mu_n), post_Gamma_ta.write(n, post_Gamma_n)]

        prior_Gamma_ta = tf.TensorArray(float_type, size=B)
        post_mu_ta = tf.TensorArray(float_type, size=B)
        post_Gamma_ta = tf.TensorArray(float_type, size=B)

        [_, _, _,
         prior_Gamma_ta, post_mu_ta, post_Gamma_ta] = \
            tf.while_loop(cond,
                          body,
                          [tf.constant(0, tf.int32),
                           mu_0,
                           Gamma_0,
                           prior_Gamma_ta,
                           post_mu_ta,
                           post_Gamma_ta])

        prior_Gamma = prior_Gamma_ta.stack()
        post_mu = post_mu_ta.stack()
        post_Gamma = post_Gamma_ta.stack()

        return prior_Gamma, post_mu, post_Gamma

    def run(self, y, Sigma_0, Omega_0, mu_0, Gamma_0, Nmax=2):
        y = np.concatenate([y.real, y.imag], axis=1)
        return self.sess.run(self.result, feed_dict={self.y_pl: y,
                                                     self.mu_0_pl: mu_0,
                                                     self.Gamma_0_pl: Gamma_0,
                                                     self.Sigma_0_pl: Sigma_0,
                                                     self.Omega_0_pl: Omega_0,
                                                     self.Nmax_pl: Nmax})
