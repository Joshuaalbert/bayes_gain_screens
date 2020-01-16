from .update import UpdatePy
from scipy.linalg import solve_triangular
from scipy.optimize import minimize
import numpy as np
import tensorflow.compat.v1 as tf
from .. import TEC_CONV


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


def constrain_clock(v):
    return constrain(v, 0.001, 5.)


def deconstrain_clock(v):
    return deconstrain(v, 0.001, 5.)


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

    def __init__(self, Yreal, Yimag, freqs, mean_prior, cov_prior,
                 sigma_real=None, sigma_imag=None):
        self.tec_conv = TEC_CONV / freqs
        self.clock_conv = (2 * np.pi * 1e-9) * freqs
        # Nf
        self.Yreal = Yreal
        self.Yimag = Yimag

        self.mean_prior = mean_prior
        self.cov_prior = cov_prior
        self.tril = np.tril_indices(3)
        self.L_param = np.zeros((3,3))
        self.L_prior = np.linalg.cholesky(cov_prior)

        self.sigma_real = sigma_real
        self.sigma_imag = sigma_imag

    # def _scalar_KL(self, mean, uncert, mean_prior, uncert_prior):
    #     # Get KL
    #     q_var = np.maximum(np.square(uncert), 1e-6)
    #     var_prior = np.maximum(np.square(uncert_prior), 1e-6)
    #     trace = q_var / var_prior
    #     mahalanobis = np.square(mean - mean_prior) / var_prior
    #     constant = -1.
    #     logdet_qcov = np.log(var_prior / q_var)
    #     twoKL = mahalanobis + constant + logdet_qcov + trace
    #     prior_KL = 0.5 * twoKL
    #     return prior_KL

    # def _corr_KL_(self, mtec, mconst, mclock, ltec, lconst, lclock, rho_tec_const, rho_tec_clock, rho_const_clock, ptec,
    #              pconst, pclock, qtec, qconst, qclock):
    #     maha = (mtec - ptec) ** 2 / qtec ** 2 + (mconst - pconst) ** 2 / qconst ** 2 + (
    #                 mclock - pclock) ** 2 / qclock ** 2
    #     trace = ltec ** 2 / qtec ** 2 + lconst ** 2 / qconst ** 2 + lclock ** 2 / qclock ** 2
    #     logdet = 2. * (np.log(qtec) + np.log(qconst) + np.log(qclock) - np.log(ltec) - np.log(lconst) - np.log(
    #         lclock) - 0.5 * np.log((1 - rho_tec_clock ** 2) * (1. - rho_tec_const ** 2) - (
    #                 rho_const_clock - rho_tec_const * rho_tec_clock) ** 2))
    #     constant = -3.
    #     return 0.5 * (maha + trace + logdet + constant)

    def _corr_KL(self, m, L, q, LQ):
        # print(m, L, q, LQ)
        maha = m - q
        maha = np.sum(np.square(solve_triangular(LQ,maha[:,None],lower=True)))
        trace = np.sum(np.square(solve_triangular(LQ,L, lower=True)))
        logdet = 2.*(np.sum(np.log(np.diag(LQ))) - np.sum(np.log(np.abs(np.diag(L)))))
        constant = -3.
        KL = 0.5*(maha+trace+logdet+constant)
        return KL

    def var_exp(self, mtec, mconst, mclock, Yreal,
                Yimag, sigma_real, sigma_imag):
        """
        Analytic int log p(y | ddtec, const, sigma^2) N[ddtec | q, Q]  N[const | f, F] dddtec dconst
        :param l:
        :param Yreal:
        :param Yimag:
        :param sigma_real:
        :param sigma_imag:
        :param k:
        :return:
        """
        a = 1. / sigma_real
        b = 1. / sigma_imag
        # T_tec_const = np.sqrt(1. - rho_tec_const ** 2)
        phi = mconst + self.tec_conv * mtec + self.clock_conv * mclock
        L00, L10, L11, L20, L21, L22 = self.L_param[self.tril]
        theta = (L00*self.tec_conv + L10 + L20*self.clock_conv)**2 + \
                (L11 + L21*self.clock_conv)**2 + \
                (L22 * self.clock_conv)**2
        # theta = (self.tec_conv * ltec + lconst * rho_tec_const + lclock * rho_tec_clock) ** 2 + \
        #         (lconst * T_tec_const + lclock * self.clock_conv * (
        #                     rho_const_clock - rho_tec_const * rho_tec_clock) / T_tec_const) ** 2 + \
        #         lclock ** 2 * self.clock_conv ** 2 * (1. - rho_tec_clock ** 2 - (
        #             rho_const_clock - rho_tec_const * rho_tec_clock) ** 2 / T_tec_const ** 2)
        res = -b ** 2 * (1. + 2. * Yimag ** 2)
        res += -a ** 2 * (1. + 2. * Yreal ** 2)
        res += 4. * (-np.log(2. * np.pi) + np.log(a) + np.log(b))
        res += np.exp(-2. * theta) * (b ** 2 - a ** 2) * np.cos(2. * phi)
        res += 4.*np.exp(-0.5 * theta) * (a ** 2 * Yreal * np.cos(phi) + b ** 2 * Yimag * np.sin(phi))
        res *= 0.25
        return np.sum(res, axis=-1)

    def loss_func(self, params):
        """
        VI loss
        :param params: tf.Tensor
            shapfrom scipy.optimize import brute, fmine [D]
        :return: tf.Tensor
            scalar The loss
        """
        tec_mean, const_mean, clock_mean = params[:3]
        self.L_param[self.tril] = params[3:]

        var_exp = self.var_exp(tec_mean, const_mean, clock_mean, self.Yreal, self.Yimag,
                               self.sigma_real, self.sigma_imag)
        # Get KL
        m = np.array([tec_mean, const_mean, clock_mean])
        KL = self._corr_KL(m, self.L_param, self.mean_prior, self.L_prior)
        # scalar
        loss = np.negative(var_exp - KL)
        return loss


class UpdateGainsToTecConstClockCorr(UpdatePy):
    """
    Uses variational inference to condition TEC on Gains.
    """

    def __init__(self, freqs, tec_scale=300., **kwargs):
        super(UpdateGainsToTecConstClockCorr, self).__init__(**kwargs)
        self.freqs = freqs
        self.tec_scale = tec_scale

    def _forward(self, samples):
        """
        Computes the data-domain samples by pushing forward.
        :param samples: tf.Tensor
            [S, B, K]
        :return: tf.Tensor
            [S, B, N]
        """

        tec_conv = tf.constant(TEC_CONV / self.freqs, samples.dtype)
        clock_conv = tf.constant((2. * np.pi * 1e-9) * self.freqs)
        # S, B, Nf
        phase = samples[:, :, 0:1] * tec_conv + samples[:, :, 1:2] + samples[:, :, 2:3] * clock_conv
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

        Nf = self.freqs.shape[0]

        sigma_real = np.sqrt(np.diag(Sigma)[:Nf])
        sigma_imag = np.sqrt(np.diag(Sigma)[Nf:])

        s = SolveLossVI(Y[:Nf], Y[Nf:], self.freqs,
                        mean_prior=prior_mu,cov_prior=prior_Gamma,
                        sigma_real=sigma_real, sigma_imag=sigma_imag)

        basin = np.mean(np.abs(np.pi / s.tec_conv)) * 0.5
        num_basin = int(self.tec_scale / basin) + 1

        x0_cov = np.eye(3)
        x0_cov[0,0] = 5.
        x0_cov[1,1] = 0.1
        x0_cov[2,2] = 0.1
        x0 = np.array(list(np.zeros(3))+list(x0_cov[s.tril]))

        res = minimize(s.loss_func,
                       x0,
                       method='BFGS').x

        obj_try = np.stack([s.loss_func([res[0] + i * basin, res[1], res[2], res[3], res[4], res[5], res[6], res[7], res[8]]) for i in
                            range(-num_basin, num_basin + 1, 1)], axis=0)
        which_basin = np.argmin(obj_try, axis=0)
        x_next = np.array([res[0] + (which_basin - float(num_basin)) * basin, res[1], res[2], res[3], res[4], res[5], res[6], res[7], res[8]])
        # last_res = res
        res = minimize(s.loss_func, x_next, method='BFGS').x

        tec_mean, const_mean, clock_mean = res[:3]
        s.L_param[s.tril] = res[3:]



        post_mu = np.array([tec_mean, const_mean, clock_mean], np.float64)
        post_cov = s.L_param.dot(s.L_param.T)


        return [post_mu, post_cov]
