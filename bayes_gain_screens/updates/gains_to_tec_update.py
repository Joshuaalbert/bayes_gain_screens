from .update import UpdatePy
from scipy.linalg import solve_triangular
from scipy.optimize import minimize, brute
import numpy as np
import tensorflow as tf
from scipy.stats import multivariate_normal


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
        #S_tec, 2*Nf
        dx = np.concatenate([Yreal_m - self.Yreal, Yimag_m - self.Yimag], axis=1)
        #S_tec
        log_prob = log_mv_gaussian_pdf(dx, self.L_Sigma)
        # S_tec -> scalar
        var_exp = np.sum(log_prob * self.w_tec, axis=0)
        # Get KL
        tec_prior_KL = self._scalar_KL(tec_mean, tec_uncert, self.tec_mean_prior, self.tec_uncert_prior)
        # scalar
        loss = np.negative(var_exp - tec_prior_KL)
        return loss


class UpdateGainsToTec(UpdatePy):
    """
    Uses variational inference to condition TEC on Gains.
    """

    def __init__(self, freqs, tec_scale=300., spacing=10., **kwargs):
        super(UpdateGainsToTec, self).__init__(**kwargs)
        self.freqs = freqs
        self.tec_scale = tec_scale
        self.spacing = spacing

    def _forward(self, samples):
        """
        Computes the data-domain samples by pushing forward.
        :param samples: tf.Tensor
            [S, B, K]
        :return: tf.Tensor
            [S, B, N]
        """

        tec_conv = tf.constant(-8.4479745e6 / self.freqs, samples.dtype)
        # S, B, Nf
        phase = samples[:, :, 0:1] * tec_conv
        return tf.concat([tf.math.cos(phase), tf.math.sin(phase)], axis=-1)

    def _update_function(self, t, prior_mu, prior_Gamma, Y, Sigma, *serve_values):
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

        ###
        # The filter expects K > 1 so we augent with dummy extra that has no effect.

        Nf = self.freqs.shape[0]

        try:
            L_Sigma = np.linalg.cholesky(Sigma + 1e-4 * np.eye(2 * Nf))
        except:
            L_Sigma = np.diag(np.sqrt(np.diag(Sigma + 1e-4 * np.eye(2 * Nf))))

        gains = Y[:Nf] + 1j * Y[Nf:]

        s = SolveLossVI(gains.real, gains.imag, self.freqs,
                        tec_mean_prior=prior_mu[0], tec_uncert_prior=np.sqrt(prior_Gamma[0, 0]),
                        S=20, L_Sigma=L_Sigma)

        sol1 = brute(lambda p: s.loss_func([p[0], deconstrain_tec(5.)]),
                     (slice(-self.tec_scale, self.tec_scale, self.spacing),))
        tec_conv = -8.4479745e6 / self.freqs
        phase_model = sol1[0]*tec_conv
        gains_model = np.exp(1j*phase_model)
        total_res = np.abs(gains - gains_model)
        keep = np.where(total_res < np.sort(total_res)[-3])[0]

        s = SolveLossVI(gains.real[keep], gains.imag[keep], self.freqs[keep],
                        tec_mean_prior=prior_mu[0], tec_uncert_prior=np.sqrt(prior_Gamma[0, 0]),
                        S=20, L_Sigma=L_Sigma)

        sol3 = minimize(s.loss_func,
                        np.array([sol1[0], deconstrain_tec(5.)]),
                        method='BFGS').x

        tec_mean = sol3[0]
        tec_uncert = constrain_tec(sol3[1])

        post_mu = np.array([tec_mean, 0], np.float64)
        post_cov = np.array([[tec_uncert ** 2, 0.], [0., 1. ** 2]], np.float64)

        return [post_mu, post_cov]
