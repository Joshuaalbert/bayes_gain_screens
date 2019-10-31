import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from scipy.ndimage import median_filter
from scipy.special import digamma
from scipy.optimize import minimize

class Smoother(object):
    def __init__(self, x, y, deg=1):
        """
        Smoothing of y over x using Bayesian polynomical regression.
        :param x: np.array
            [N] the coordinates
        :param y: np.array
            [M, N] the data
            Assumption A: y[i, j] = p[i,0] * x[j]**deg + ... + p[i, deg]
            y[i, j] = X[j, k].beta[i, k]
            X[j, k] = x[j]**(deg - k)
            beta[i,k] = p[i,k]
            Assumption B: y[i+1,:] = y[i,:] + N[0, C]
        :param deg: int
            The order of fit.
        """
        sigma, p = self.non_bayes_fit(x, y, deg, filter_size=1)
        mu0, Lambda0, a0, b0 = self.prior_fit(sigma, p)
        #M, N, deg+1
        X = x[:, None]**(deg - np.arange(deg+1)[None, :])



    def prior_fit(self,sigma, p):
        #M, deg+1
        P = np.stack(p, axis=1)
        #deg+1
        mu0 = np.mean(P, axis=0)
        #deg+1, deg+1
        Lambda0 = np.linalg.pinv(np.cov(P, rowvar=False)/np.mean(sigma**2))

        tau2 = np.size(sigma)*np.sum(1./sigma**2)
        def nu_loss(params):
            nu = params[0]
            lhs = np.log(nu/2.) + digamma(nu/2.)
            rhs = np.sum(2.*np.log(sigma)) - sigma.size * np.log(tau2)
            return np.abs(lhs - rhs)
        nu = minimize(nu_loss, [2.*np.mean(sigma**2)/(np.mean(sigma**2) - tau2)], method='BFGS').x
        a0 = nu/2.
        b0 = nu * tau2 / 2.

        return mu0, Lambda0, a0, b0

    def non_bayes_fit(self, x, y, deg=1, filter_size=1):
        """

        :param x: np.array
            [N] the coordinates
        :param y: np.array
            [M, N] the data
            Assumption A: y[i, j] = p[i,0] * x[j]**deg + ... + p[i, deg]
            Assumption B: y[i+1,:] = y[i,:] + N[0, C]
        :param deg: int
            The order of fit.
        :return:
        """
        y0 = np.copy(y)
        for i in range(3):
            #M, N
            y_updated = sum([median_filter(p, filter_size) * x[:, None] ** (deg - i) for i, p in
                 enumerate(np.polyfit(x, y0.T, deg=deg))]).T
            #M, N
            res = y - y_updated
            #N
            sigma = np.sqrt(np.mean(res ** 2, axis=0))
            #M, N
            keep = np.abs(res) < 3. * sigma[:, None]
            where_over_flagged = np.where(np.sum(keep, axis=-1) < keep.shape[1] / 2.)[0]
            for t in where_over_flagged:
                keep[t, :] = np.abs(res[:, t]) <= np.median(np.abs(res[t, :]))
                if np.sum(keep[t, :]) < keep.shape[1] / 2.:
                    keep[t, :] = True
            y0 = np.where(keep, y, y_updated)

        p = np.polyfit(x, y0.T, deg=deg)
        # M, N
        y_updated = sum([median_filter(p, filter_size) * x[:, None] ** (deg - i) for i, p in
                         enumerate(p)]).T

        return sigma, p

