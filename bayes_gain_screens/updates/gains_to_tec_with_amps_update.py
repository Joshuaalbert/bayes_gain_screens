from .update import UpdatePy
from scipy.optimize import minimize
import jax.numpy as np
import jax
from jax.scipy.linalg import solve_triangular
import tensorflow.compat.v1 as tf
from scipy.stats import multivariate_normal
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

    def __init__(self, amps, Yreal, Yimag, freqs, tec_mean_prior=0., tec_uncert_prior=100.,
                 sigma_real=None, sigma_imag=None):
        self.amps = amps
        self.tec_conv = TEC_CONV / freqs
        # Nf
        self.Yreal = Yreal
        self.Yimag = Yimag

        self.tec_mean_prior = tec_mean_prior
        self.tec_uncert_prior = tec_uncert_prior

        self.sigma_real = sigma_real
        self.sigma_imag = sigma_imag

    def scalar_KL(self, mean, uncert, mean_prior, uncert_prior):
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

    def var_exp(self, m, l, Yreal, Yimag, sigma_real, sigma_imag):
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
        a = 1./sigma_real
        b = 1./sigma_imag
        phi = self.tec_conv * m
        theta = self.tec_conv ** 2 * l * l
        res = -b**2 * (self.amps**2 + 2. * Yimag ** 2)
        res += -a**2 * (self.amps**2 + 2. * Yreal**2)
        res += -4.*np.log(2.*np.pi/(a*b))
        res += self.amps*np.exp(-2. * theta) * (self.amps*(b**2 - a**2) * np.cos(2. * phi) + 4.* np.exp(1.5*theta) * (a**2 * Yreal * np.cos(phi) + b**2 * Yimag * np.sin(phi)))
        res *= 0.25
        return np.sum(res, axis=-1)

    def loss_func(self, params):
        tec_mean, _tec_uncert = params[0], params[1]
        tec_uncert = constrain_tec(_tec_uncert)
        _var_exp = self.var_exp(tec_mean, tec_uncert, self.Yreal, self.Yimag, self.sigma_real, self.sigma_imag)
        # Get KL
        tec_prior_KL = self.scalar_KL(tec_mean, tec_uncert, self.tec_mean_prior, self.tec_uncert_prior)
        # scalar
        loss = tec_prior_KL - _var_exp
        return loss

class UpdateGainsToTecAmps(UpdatePy):
    """
    Uses variational inference to condition TEC on Gains.
    """

    def __init__(self, freqs, tec_scale=300., spacing=10., **kwargs):
        super(UpdateGainsToTecAmps, self).__init__(**kwargs)
        self.freqs = freqs
        self.tec_scale = tec_scale
        self.spacing = spacing

    def _forward(self, samples, *serve_values):
        """
        Computes the data-domain samples by pushing forward.
        :param samples: tf.Tensor
            [S, B, K]
        :return: tf.Tensor
            [S, B, N]
        """
        amps = serve_values[0]

        tec_conv = tf.constant(TEC_CONV / self.freqs, samples.dtype)
        # S, B, Nf
        phase = samples[:, :, 0:1] * tec_conv
        return tf.concat([amps*tf.math.cos(phase), amps*tf.math.sin(phase)], axis=-1)

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
        amps = serve_values[0]

        Nf = self.freqs.shape[0]

        # try:
        #     L_Sigma = np.linalg.cholesky(Sigma + 1e-4 * np.eye(2 * Nf))
        # except:
        #     L_Sigma = np.diag(np.sqrt(np.diag(Sigma + 1e-4 * np.eye(2 * Nf))))

        sigma_real = np.sqrt(np.diag(Sigma)[:Nf])
        sigma_imag = np.sqrt(np.diag(Sigma)[Nf:])

        s = SolveLossVI(amps, Y[:Nf], Y[Nf:], self.freqs,
                        tec_mean_prior=prior_mu[0], tec_uncert_prior=np.sqrt(prior_Gamma[0, 0]),
                        sigma_real=sigma_real, sigma_imag = sigma_imag)

        basin = np.mean(np.abs(np.pi / s.tec_conv))*0.5
        num_basin = int(self.tec_scale/basin)+1

        @jax.jit
        def value_and_grad(x):
            f,j = jax.value_and_grad(s.loss_func)(x)
            return f, np.array(j[0])

        res = minimize(value_and_grad,
                       np.array([prior_mu[0], deconstrain_tec(5.)]),
                       jac=True,
                       method='BFGS').x
        obj_try = np.stack([s.loss_func([res[0] + i * basin, res[1]]) for i in range(-num_basin, num_basin+1, 1)], axis=0)
        which_basin = np.argmin(obj_try, axis=0)
        x_next = np.array([res[0] + (which_basin - float(num_basin)) * basin, res[1]])
        sol = minimize(value_and_grad, x_next, jac=True, method='BFGS').x

        tec_mean = sol[0]
        tec_uncert = constrain_tec(sol[1])

        post_mu = np.array([tec_mean, 0], np.float64)
        post_cov = np.array([[tec_uncert ** 2, 0.], [0., 1. ** 2]], np.float64)

        return [post_mu, post_cov]

def speed():
    import numpy as onp
    onp.random.seed(0)
    freqs = np.linspace(121e6, 166e6, 24)
    tec_true = 87.
    phase_true = tec_true * TEC_CONV / freqs
    Sigma = 0.2 ** 2 * np.eye(freqs.size * 2)
    Y_obs = np.exp(1j * phase_true) + 0.2 * (onp.random.normal(size=phase_true.shape) + 1j * onp.random.normal(size=phase_true.shape))
    amp = np.ones(freqs.size)

    model = UpdateGainsToTecAmps(freqs, tec_scale=300., spacing=10.)

    res = model._update_function(0, np.zeros(2), 100**2*np.eye(2), Y_obs, Sigma, amp)
    print(res)

if __name__=='__main__':
    speed()