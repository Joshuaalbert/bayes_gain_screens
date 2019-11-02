import os

os.environ['OMP_NUM_THREADS'] = "1"
import numpy as np
import pylab as plt
from matplotlib.mlab import griddata
from scipy.optimize import brute, fmin, minimize
from scipy.ndimage import median_filter
from scipy.special import k0, k0e
from scipy.interpolate import Rbf
from scipy.linalg import solve_triangular
from bayes_gain_screens import logging
from bayes_gain_screens.datapack import DataPack, update_h5parm
from bayes_gain_screens.misc import make_soltab, make_coord_array
from dask.multiprocessing import get
from collections import deque
from scipy.optimize import least_squares
from scipy.interpolate import interp1d

"""
This script is still being debugged/tested. 
Get's TEC from gains.
"""

MIN_UNCERT = 1e-3


def constrain(v, a, b):
    return a + (np.tanh(v) + 1) * (b - a) / 2.


def deconstrain(v, a, b):
    return np.arctanh(np.clip((v - a) * 2. / (b - a) - 1., -0.999, 0.999))


def wrap(p):
    return np.arctan2(np.sin(p), np.cos(p))


def log_k0(x):
    """
    ke0(x) = exp(x) k0(x)
    log(k0(x)) = log(k0e(x)) - x
    :param x:
    :return:
    """
    # if np.any(np.where(k0e(x) < 1e-10)):
    #     print(np.where(k0e(x) < 1e-10))
    return np.log(k0e(x)) - x


def log_laplace_pdf(x, y, b1, b2, rho):
    # S, Nf
    x = x / b1
    y = y / b2

    log_prob = log_k0(np.sqrt(2. * (x ** 2 - 2. * rho * x * y + y ** 2) / (1. - rho ** 2)))
    if ~np.all(np.isfinite(log_prob)):
        raise ValueError("Invalid log_prob {} x {} y {} laplace {} {} {}".format(log_prob, x, y, b1, b2, rho))
    log_prob += -np.log(np.pi * b1 * b2 * np.sqrt(1. - rho ** 2))
    if ~np.all(np.isfinite(log_prob)):
        raise ValueError("Invalid log_prob {} laplace {} {} {}".format(log_prob, b1, b2, rho))

    return log_prob

def log_gaussian_pdf(x, y, sigma):
    # S, Nf
    x = x / sigma
    y = y / sigma

    log_prob = -(x ** 2 + y ** 2) / 2.
    if ~np.all(np.isfinite(log_prob)):
        raise ValueError("Invalid log_prob {} x {} y {} laplace {} {} {}".format(log_prob, x, y, sigma))
    log_prob += -np.log(2*np.pi) - 2*np.log(sigma)
    if ~np.all(np.isfinite(log_prob)):
        raise ValueError("Invalid log_prob {} laplace {} {} {}".format(log_prob,sigma))

    return log_prob


def log_laplace_uncorr_pdf(x, y, b):
    x = x / b
    y = y / b
    log_prob = np.log(k0(np.sqrt(2. * (x ** 2 + y ** 2))))
    log_prob += -np.log(np.pi * b ** 2)
    return log_prob


class SolveLossMaxLike(object):
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

    def __init__(self, Yreal, Yimag, freqs, laplace_b=None,sigma=None):

        self.tec_conv = -8.4479745e6 / freqs
        # Nf
        self.amp = np.sqrt(np.square(Yreal) + np.square(Yimag))
        self.Yreal = Yreal
        self.Yimag = Yimag

        self.phase = wrap(np.arctan2(Yimag, Yreal))

        self.phase_model = lambda params: self.tec_conv * params[0] + params[1]
        self.laplace_b = laplace_b
        self.sigma = sigma

    def loss_func(self, params):
        """
        non-VI loss
        """
        tec, const = params[0], params[1]

        # Nf
        phase = tec * self.tec_conv + const
        Yreal_m = self.amp * np.cos(phase)
        Yimag_m = self.amp * np.sin(phase)

        # Nf
        # log_prob = log_laplace_pdf((Yreal_m - self.Yreal),
        #                            (Yimag_m - self.Yimag),
        #                            self.laplace_b[0, :],
        #                            self.laplace_b[1, :],
        #                            self.laplace_b[2, :])
        log_prob = log_gaussian_pdf((Yreal_m - self.Yreal),
                                   (Yimag_m - self.Yimag),
                                    self.sigma)
        # inf_mask = np.isinf(log_prob)
        # # Nf
        # scale = inf_mask.size / (inf_mask.size - inf_mask.sum())
        # log_prob = np.where(inf_mask, 0., log_prob * scale)

        log_prob = np.sum(log_prob)

        loss = np.negative(log_prob)
        if ~np.isfinite(loss):
            raise ValueError("Invalid loss {} vparams {} laplace {}".format(loss, params, self.laplace_b))
        if ~np.all(np.isfinite(params)):
            raise ValueError("Invalid params: {}".format(params))
        # B
        return loss


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
                 S=20, laplace_b=None, sigma=None):
        self.x, self.w = np.polynomial.hermite.hermgauss(S)
        self.x_tec = self.x
        self.x_const = self.x
        self.w /= np.pi
        self.w_tec = self.w
        self.w_const = self.w

        self.tec_conv = -8.4479745e6 / freqs
        # Nf
        self.amp = np.sqrt(np.square(Yreal) + np.square(Yimag))
        self.Yreal = Yreal
        self.Yimag = Yimag

        self.phase = wrap(np.arctan2(Yimag, Yreal))

        self.tec_mean_prior = tec_mean_prior
        self.tec_uncert_prior = tec_uncert_prior
        self.const_mean_prior = const_mean_prior
        self.const_uncert_prior = const_uncert_prior

        self.phase_model = lambda params: self.tec_conv * params[0] + params[2]
        self.laplace_b = laplace_b
        self.sigma = sigma

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

    def loss_func_log(self, params):
        """
        VI loss
        :param params: tf.Tensor
            shapfrom scipy.optimize import brute, fmine [D]
        :return: tf.Tensor
            scalar The loss
        """
        tec_mean, log_tec_uncert, const_mean, log_const_uncert = params[0], params[1], params[2], params[3]

        tec_uncert = MIN_UNCERT + np.exp(log_tec_uncert)
        const_uncert = MIN_UNCERT + np.exp(log_const_uncert)

        # S_tec
        tec = tec_mean + np.sqrt(2.) * tec_uncert * self.x_tec
        # S_const
        const = const_mean + np.sqrt(2.) * const_uncert * self.x_const
        # S_tec, S_const, Nf
        phase = tec[:, None, None] * self.tec_conv + const[:, None]
        Yreal_m = self.amp * np.cos(phase)
        Yimag_m = self.amp * np.sin(phase)

        # S_tec, S_const, Nf
        log_prob = log_laplace_pdf((Yreal_m - self.Yreal),
                                   (Yimag_m - self.Yimag),
                                   self.laplace_b[0, :],
                                   self.laplace_b[1, :],
                                   self.laplace_b[2, :])
        # inf_mask = np.isinf(log_prob)
        # Nf
        # scale = inf_mask.size/(inf_mask.size - inf_mask.sum(0).sum(0))
        # log_prob = np.where(inf_mask, 0., log_prob)

        # S_tec, S_const
        log_prob = np.sum(log_prob, axis=-1)
        # scalar
        var_exp = np.sum(np.sum(log_prob * self.w_tec[:, None], axis=0) * self.w_const, axis=0)

        # Get KL
        tec_prior_KL = self._scalar_KL(tec_mean, tec_uncert, self.tec_mean_prior, self.tec_uncert_prior)
        const_prior_KL = self._scalar_KL(const_mean, const_uncert, self.const_mean_prior, self.const_uncert_prior)
        loss = np.negative(var_exp - tec_prior_KL - const_prior_KL)
        if ~np.isfinite(loss):
            raise ValueError(
                "Invalid loss {} var exp {} tec KL {} const KL {} params {} laplace {}".format(loss, var_exp,
                                                                                               tec_prior_KL,
                                                                                               const_prior_KL, params,
                                                                                               self.laplace_b))
        if ~np.all(np.isfinite(params)):
            raise ValueError("Invalid params: {}".format(params))
        # B
        return loss

    def loss_func(self, params):
        """
        VI loss
        :param params: tf.Tensor
            shapfrom scipy.optimize import brute, fmine [D]
        :return: tf.Tensor
            scalar The loss
        """
        tec_mean, _tec_uncert, const_mean, _const_uncert = params[0], params[1], params[2], params[3]

        tec_uncert = constrain(_tec_uncert, 0.01, 55.)
        const_uncert = constrain(_const_uncert, 0.001, 2 * np.pi)

        # S_tec
        tec = tec_mean + np.sqrt(2.) * tec_uncert * self.x_tec
        # S_const
        const = const_mean + np.sqrt(2.) * const_uncert * self.x_const
        # S_tec, S_const, Nf
        phase = tec[:, None, None] * self.tec_conv + const[:, None]
        Yreal_m = self.amp * np.cos(phase)
        Yimag_m = self.amp * np.sin(phase)

        # S_tec, S_const, Nf
        # log_prob = log_laplace_pdf((Yreal_m - self.Yreal),
        #                            (Yimag_m - self.Yimag),
        #                            self.laplace_b[0, :],
        #                            self.laplace_b[1, :],
        #                            self.laplace_b[2, :])
        log_prob = log_gaussian_pdf((Yreal_m - self.Yreal),
                                    (Yimag_m - self.Yimag),
                                    self.sigma)
        # inf_mask = np.isinf(log_prob)
        # Nf
        # scale = inf_mask.size/(inf_mask.size - inf_mask.sum(0).sum(0))
        # log_prob = np.where(inf_mask, 0., log_prob)

        # S_tec, S_const
        log_prob = np.sum(log_prob, axis=-1)
        # scalar
        var_exp = np.sum(np.sum(log_prob * self.w_tec[:, None], axis=0) * self.w_const, axis=0)

        # Get KL
        tec_prior_KL = self._scalar_KL(tec_mean, tec_uncert, self.tec_mean_prior, self.tec_uncert_prior)
        const_prior_KL = self._scalar_KL(const_mean, const_uncert, self.const_mean_prior, self.const_uncert_prior)
        loss = np.negative(var_exp - tec_prior_KL - const_prior_KL)
        if ~np.isfinite(loss):
            raise ValueError(
                "Invalid loss {} var exp {} tec KL {} const KL {} params {} laplace {}".format(loss, var_exp,
                                                                                               tec_prior_KL,
                                                                                               const_prior_KL, params,
                                                                                               self.laplace_b))
        if ~np.all(np.isfinite(params)):
            raise ValueError("Invalid params: {}".format(params))
        # B
        return loss


def sequential_solve(Yreal, Yimag, freqs, debug_indicator=None, output_dir=None, unravel_index=None):
    """
    Run on blocks of time.

    :param Yreal:
        [D, Nf, Nt]
    :param Yimag:
    :param freqs:
    :return:
        [D, Nt], [D, Nt]
    """

    if debug_indicator is not None and output_dir is None:
        raise ValueError("output dir not specified")

    if debug_indicator is not None:
        # debug stuff
        np.random.seed(0)
        debug_tec_range = np.random.uniform(-200, 200, size=15000)
        debug_const_range = np.random.uniform(-np.pi, np.pi, size=15000)
        debug_grid_points = np.linspace(-200., 200., 401), np.linspace(-np.pi, np.pi, 100)
        if not callable(debug_indicator) and isinstance(debug_indicator, bool):
            debug_indicator = lambda *x: debug_indicator

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    priors = dict(tec_mean_prior=0., tec_uncert_prior=100., const_mean_prior=0., const_uncert_prior=1.)

    D, Nf, N = Yreal.shape

    tec_mean_array = np.zeros((D, N))
    tec_uncert_array = np.zeros((D, N))
    const_mean_array = np.zeros((D, N))
    const_uncert_array = np.zeros((D, N))

    for d in range(D):
        dir_idx, ant_idx = unravel_index[0][d], unravel_index[1][d]
        # # 3, Nf
        # laplace_b = get_residual_laplace_statistics(Yimag,
        #                                             Yreal,
        #                                             ant_idx, d, dir_idx, output_dir)
        # laplace_b[0,:] = np.clip(laplace_b[0,:], 0.001, 1.)
        # laplace_b[1,:] = np.clip(laplace_b[1,:], 0.001, 1.)

        # Nf, Nf N, Nf N
        sigma, Yreal_res, Yimag_res = get_residual_gaussian_statistics(Yimag,
                                                Yreal,
                                                ant_idx, d, dir_idx, output_dir)

        tec_diff_prior = 50.
        const_diff_prior = 0.2

        # flag based on smoothed residuals
        total_res = np.sqrt(Yreal_res ** 2 + Yimag_res ** 2)
        weights = total_res < 3.*sigma[:, None]#np.sort(total_res, axis=0)[-3, :]
        with np.printoptions(precision=2):
            logging.info("Average flagging per channel:\n{}".format(1. - weights.mean(1)))
        where_over_flagged = np.where(weights.sum(0) < weights.shape[0]/2.)[0]
        for t in where_over_flagged:
            logging.info("Time step {} over flagged".format(t))
            weights[:,t] = total_res[:,t] < np.sort(total_res[:,t])[-weights.shape[0]//2]

        for n in range(N):

            keep = np.where(weights[:, n])[0]

            # least square two pass
            ls_obj = SolveLossMaxLike(Yreal[d, keep, n], Yimag[d, keep, n], freqs[keep], sigma=sigma[keep])
            result1 = brute(ls_obj.loss_func,
                            (slice(-200., 200., 5.),
                             slice(-np.pi, np.pi, 0.05)),
                            finish=least_squares)
            result1[1] = np.arctan2(np.sin(result1[1]), np.cos(result1[1]))
            ls_obj = SolveLossVI(Yreal[d, keep, n], Yimag[d, keep, n], freqs[keep], sigma=sigma[keep],**priors)
            result2 = minimize(ls_obj.loss_func,[result1[0], deconstrain(3., 0.01, 55.), result1[1], deconstrain(0.5, 0.001, 2 * np.pi)],
                               method='BFGS').x
            # least_squares(ls_obj.loss_func,[result1[0], deconstrain(3., 0.01, 55.), result1[1], deconstrain(0.5, 0.001, 2 * np.pi)], method='lm').x
            result2[2] = np.arctan2(np.sin(result2[2]), np.cos(result2[2]))
            print(result1, result2)
            tec_mean, _tec_uncert, const_mean, _const_uncert = result2
            tec_uncert = constrain(_tec_uncert, 0.01, 55.)
            const_uncert = constrain(_const_uncert, 0.001, 2 * np.pi)
            tec_mean_array[d, n] = tec_mean
            tec_uncert_array[d, n] = tec_uncert
            const_mean_array[d, n] = const_mean
            const_uncert_array[d, n] = const_uncert

            # if n > 20:
            #     tec_diff_prior = np.maximum(np.sqrt(np.mean(np.square(np.diff(tec_mean_array[d, :n+1])))), 10.)
            priors['tec_mean_prior'] = tec_mean
            priors['tec_uncert_prior'] = np.sqrt(tec_diff_prior ** 2 + tec_uncert ** 2)
            priors['const_mean_prior'] = const_mean
            priors['const_uncert_prior'] = np.sqrt(const_diff_prior ** 2 + tec_uncert ** 2)
            with np.printoptions(precision=2):
                logging.info(
                    "dir {:02d} ant {:02d} time {:03d} tec {} +- {} const {} +- {}".format(dir_idx, ant_idx, n, tec_mean,
                                                                                           tec_uncert, const_mean,
                                                                                           const_uncert))

            if debug_indicator is not None:
                if debug_indicator(d, n):
                    ls_obj = SolveLossMaxLike(Yreal[d, :, n], Yimag[d, :, n], freqs,
                                              sigma=sigma)

                    logging.info("Plotting {} {}".format(d, n))
                    debug_loss = np.array([ls_obj.loss_func([t, c]) for t, c in
                                           zip(debug_tec_range, debug_const_range)])
                    debug_grid_loss = griddata(debug_tec_range, debug_const_range, debug_loss, *debug_grid_points,
                                               interp='linear')

                    fig, axs = plt.subplots(2, 1, figsize=(6, 6))
                    img = axs[0].imshow(debug_grid_loss, origin='lower', aspect='auto',
                                        extent=(debug_grid_points[0].min(), debug_grid_points[0].max(),
                                                debug_grid_points[1].min(), debug_grid_points[1].max()))
                    # plt.colorbar(img)
                    axs[0].set_ylabel('const [rad]')
                    axs[0].set_xlabel('tec [mTECU]')
                    xlim = axs[0].get_xlim()
                    ylim = axs[0].get_ylim()
                    axs[0].scatter(np.clip(result1[0], *xlim), np.clip(result1[1], *ylim), c='red')
                    axs[0].scatter(np.clip(result2[0], *xlim), np.clip(result2[2], *ylim), c='green')
                    axs[0].set_xlim(*xlim)
                    axs[0].set_ylim(*ylim)
                    axs[0].set_title(
                        "Solved {:.1f} +- {:.1f} mTECU const {:.2f} +- {:.2f} rad".format(tec_mean, tec_uncert,
                                                                                          const_mean, const_uncert))
                    axs[1].plot(freqs, wrap(ls_obj.phase_model(result1)), c='red', label='mean phase ML')
                    axs[1].plot(freqs, wrap(ls_obj.phase_model(result2[::2])), c='green', label='mean phase VI')
                    # axs[1].plot(freqs, wrap(ls_obj.phase_model([result[0] + result[1], None, const])), c='black', ls='dotted',
                    #             label=r'$+\sigma$ phase')
                    # axs[1].plot(freqs, wrap(ls_obj.phase_model([result[0] - result[1], None, const])), c='black', ls='dashed',
                    #             label=r'$-\sigma$ phase')
                    axs[1].scatter(freqs, ls_obj.phase, label='data')
                    axs[1].set_xlabel('freq (Hz)')
                    axs[1].set_ylabel('phase (rad)')
                    os.makedirs(os.path.join(output_dir, 'loss'), exist_ok=True)
                    plt.savefig(os.path.join(output_dir, 'loss',
                                             'loss_dir{:02d}_ant{:02d}_time{:03d}.png'.format(dir_idx, ant_idx, n)))
                    plt.close('all')

    return tec_mean_array, tec_uncert_array, const_mean_array, const_uncert_array


def get_residual_gaussian_statistics(Yimag, Yreal, ant_idx, d, dir_idx, output_dir):
    logging.info("Using Rbf to get noise levels for dir {:02d} ant {:02d}.".format(dir_idx, ant_idx))

    hist_dir = os.path.join(output_dir, 'histograms')
    os.makedirs(hist_dir, exist_ok=True)
    _, Nf, N = Yimag.shape
    _times = np.linspace(-1., 1., N)
    _freqs = np.linspace(-1., 1., Nf)

    Yreal = Yreal[d, :, :]
    Yimag = Yimag[d, :, :]

    Yreal_smooth, Yimag_smooth = smooth_gains(Yreal, Yimag, filter_size=1, deg=2)

    Yreal_res = Yreal - Yreal_smooth
    Yimag_res = Yimag - Yimag_smooth

    sigma = np.sqrt(np.mean(Yreal_res ** 2 + Yimag_res ** 2, axis=-1))
    with np.printoptions(precision=2):
        logging.info("dir {:02d} ant {:02d}: Found Gaussian sigma {}".format(dir_idx, ant_idx, sigma))
    return sigma, Yreal_res, Yimag_res


def get_residual_laplace_statistics(Yimag, Yreal, ant_idx, d, dir_idx, output_dir):
    logging.info("Using Rbf to get noise levels for dir {:02d} ant {:02d}.".format(dir_idx, ant_idx))

    hist_dir = os.path.join(output_dir, 'histograms')
    os.makedirs(hist_dir, exist_ok=True)
    _, Nf, N = Yimag.shape
    _times = np.linspace(-1., 1., N)
    _freqs = np.linspace(-1., 1., Nf)

    Yreal = Yreal[d, :, :]
    Yimag = Yimag[d, :, :]

    Yreal_smooth, Yimag_smooth = smooth_gains(Yreal, Yimag, filter_size=3, deg=2)

    Yreal_res = Yreal - Yreal_smooth
    Yimag_res = Yimag - Yimag_smooth

    logging.info("Fitting histograms per frequency.")
    laplace_b_freq = []
    for nu in range(Nf):
        H, binsx, binsy = np.histogram2d(Yreal_res[nu, :], Yimag_res[nu, :],
                                         bins=np.ceil(np.sqrt(N)),
                                         density=True)

        binsxc = 0.5 * (binsx[1:] + binsx[:-1])
        binsyc = 0.5 * (binsy[1:] + binsy[:-1])

        def _laplace_logpdf(log_b1, log_b2, arctanh_rho):
            b1 = np.exp(log_b1)
            b2 = np.exp(log_b2)
            rho = 0.99 * np.tanh(arctanh_rho)
            return log_laplace_pdf(binsxc[:, None], binsyc[None, :], b1, b2, rho)

        def _laplace_loss(params):
            log_b1, log_b2, arctanh_rho = params
            return np.sum(np.abs(np.exp(_laplace_logpdf(log_b1, log_b2, arctanh_rho)) - H))

        results = brute(_laplace_loss,
                        (slice(np.log(0.01), np.log(1.), np.log(100.) / 10.),
                         slice(np.log(0.01), np.log(1.), np.log(100.) / 10.),
                         slice(np.arctanh(-0.95), np.arctan(0.95),
                               (np.arctan(0.95) - np.arctan(-0.95)) / 10.)))

        log_b1, log_b2, arctanh_rho = results
        b1 = np.exp(log_b1)
        b2 = np.exp(log_b2)
        rho = 0.99 * np.tanh(arctanh_rho)
        # b1, b2, rho = [0.1, 0.1, 0.]

        logging.info(
            "freq {:02d} dir {:02d} ant {:02d}: Found laplace scale {:.3f} {:.3f} {:.3f}".format(nu, dir_idx, ant_idx,
                                                                                                 b1, b2, rho))
        laplace_b_freq.append([b1, b2, rho])

        f, axs = plt.subplots(2, 1, figsize=(4, 8))
        axs[0].imshow(H.T, extent=(binsxc.min(), binsxc.max(),
                                   binsyc.min(), binsyc.max()), vmin=0., vmax=H.max(), origin='lower',
                      aspect='auto')

        axs[1].imshow(np.exp(_laplace_logpdf(log_b1, log_b2, arctanh_rho).T), extent=(binsxc.min(), binsxc.max(),
                                                                                      binsyc.min(), binsyc.max()),
                      vmin=0.,
                      vmax=H.max(), origin='lower', aspect='auto')
        axs[0].set_title("b1 {:.2f} b2 {:.2f} rho {:.2f}".format(b1, b2, rho))
        plt.tight_layout()
        plt.savefig(os.path.join(hist_dir, 'histogram_dir{:02d}_ant{:02d}_freq{:02d}.png'.format(dir_idx, ant_idx, nu)))
        plt.close('all')
    # 3, Nf
    return np.stack(laplace_b_freq, axis=1)


def wrap(p):
    return np.arctan2(np.sin(p), np.cos(p))


def smoothamps(amps):
    freqkernel = 3
    timekernel = 31
    idxh = np.where(amps > 5.)
    idxl = np.where(amps < 0.15)
    median = np.tile(np.nanmedian(amps, axis=-1, keepdims=True), (1, 1, 1, 1, amps.shape[-1]))
    amps[idxh] = median[idxh]
    amps[idxl] = median[idxl]
    ampssmoothed = np.exp((median_filter(np.log(amps), size=(1, 1, 1, freqkernel, timekernel), mode='reflect')))
    return ampssmoothed


def smooth_gains(Yreal, Yimag, filter_size=3, deg=1):
    _Yreal, _Yimag = np.copy(Yreal), np.copy(Yimag)
    Nf, N = Yreal.shape
    _freqs = np.linspace(-1., 1., Nf)
    real = sum([median_filter(p, filter_size) * _freqs[:, None] ** (deg - i) for i, p in
                enumerate(np.polyfit(_freqs, _Yreal, deg=deg))])
    imag = sum([median_filter(p, filter_size) * _freqs[:, None] ** (deg - i) for i, p in
                enumerate(np.polyfit(_freqs, _Yimag, deg=deg))])

    res_real = np.abs(Yreal - real)
    flag = res_real > np.sort(res_real, axis=0)[-2]
    _Yreal[flag] = real[flag]
    res_imag = np.abs(Yimag - imag)
    flag = res_imag > np.sort(res_imag, axis=0)[-2]
    _Yimag[flag] = imag[flag]

    real = sum([median_filter(p, filter_size) * _freqs[:, None] ** (deg - i) for i, p in
                enumerate(np.polyfit(_freqs, _Yreal, deg=deg))])
    imag = sum([median_filter(p, filter_size) * _freqs[:, None] ** (deg - i) for i, p in
                enumerate(np.polyfit(_freqs, _Yimag, deg=deg))])

    _Yreal, _Yimag = np.copy(Yreal), np.copy(Yimag)

    res_real = np.abs(Yreal - real)
    flag = res_real > np.sort(res_real, axis=0)[-2]
    _Yreal[flag] = real[flag]
    res_imag = np.abs(Yimag - imag)
    flag = res_imag > np.sort(res_imag, axis=0)[-2]
    _Yimag[flag] = imag[flag]

    real = sum([median_filter(p, filter_size) * _freqs[:, None] ** (deg - i) for i, p in
                enumerate(np.polyfit(_freqs, _Yreal, deg=deg))])
    imag = sum([median_filter(p, filter_size) * _freqs[:, None] ** (deg - i) for i, p in
                enumerate(np.polyfit(_freqs, _Yimag, deg=deg))])

    return real, imag


def distribute_solves(datapack=None, solset='sol000', ref_dir_idx=0, num_processes=64, select={'pol': slice(0, 1, 1)},
                      plot=False, output_folder='./tec_solve', debug=False):
    output_folder = os.path.abspath(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    logging.info("Using output directory: {}".format(output_folder))

    datapack = DataPack(datapack, readonly=False)
    logging.info("Creating directionally_referenced/tec000")
    make_soltab(datapack, from_solset=solset, to_solset='directionally_referenced', from_soltab='phase000',
                to_soltab=['tec000', 'const000'])
    logging.info("Getting phase and amplitude data")
    datapack.current_solset = solset
    datapack.select(**select)
    axes = datapack.axes_phase
    antenna_labels, antennas = datapack.get_antennas(axes['ant'])
    patch_names, directions = datapack.get_directions(axes['dir'])
    timestamps, times = datapack.get_times(axes['time'])
    freq_labels, freqs = datapack.get_freqs(axes['freq'])
    pol_labels, pols = datapack.get_pols(axes['pol'])
    Npol, Nd, Na, Nf, Nt = len(pols), len(directions), len(antennas), len(freqs), len(times)
    phase_raw, axes = datapack.phase
    amp_raw, axes = datapack.amplitude

    logging.info("Smoothing amplitudes in time and frequency.")
    amp_smooth = smoothamps(amp_raw)
    logging.info("Constructing gains.")
    # Npol, Nd, Na, Nf, Nt
    Yimag_full = amp_smooth * np.sin(phase_raw)
    Yreal_full = amp_smooth * np.cos(phase_raw)
    # Nf, Npol*Nd*Na*Nt
    Yimag_full = Yimag_full.transpose((3, 0, 1, 2, 4)).reshape((Nf, -1))
    Yreal_full = Yreal_full.transpose((3, 0, 1, 2, 4)).reshape((Nf, -1))
    logging.info("Smoothing gains.")
    Yreal_full, Yimag_full = smooth_gains(Yreal_full, Yimag_full, 3, 2)
    # Npol, Nd, Na, Nf, Nt
    Yreal_full = Yreal_full.reshape((Nf, Npol, Nd, Na, Nt)).transpose((1, 2, 3, 0, 4))
    Yimag_full = Yimag_full.reshape((Nf, Npol, Nd, Na, Nt)).transpose((1, 2, 3, 0, 4))
    phase_smooth = np.arctan2(Yimag_full, Yreal_full)
    logging.info("Directionally referencing phase.")
    logging.info("Using ref_dir_idx {}".format(ref_dir_idx))
    phase_di = phase_smooth[:, ref_dir_idx:ref_dir_idx + 1, ...]
    phase_raw = phase_raw - phase_di

    # Npol, Nd, Na, Nf, Nt
    Yimag_full = amp_smooth * np.sin(phase_raw)
    Yreal_full = amp_smooth * np.cos(phase_raw)

    Yimag_full = Yimag_full.reshape((-1, Nf, Nt))
    Yreal_full = Yreal_full.reshape((-1, Nf, Nt))

    if debug:
        logging.info("Debugging on.")
        debug_indicator = lambda d, n: n % 50 == 0
    else:
        debug_indicator = lambda *x: False

    logging.info("Creating dask.")
    D = Yimag_full.shape[0]

    def d_map(start, stop):
        return np.unravel_index(range(start, stop), (Nd, Na))

    num_processes = min(D, num_processes)
    dsk = {}
    for i in range(0, D, D // num_processes):
        start = i
        stop = min(i + (D // num_processes), D)
        dsk[str(i)] = (
            sequential_solve, Yreal_full[start:stop, :, :], Yimag_full[start:stop, :, :], freqs, debug_indicator,
            output_folder, d_map(start, stop))
    logging.info("Running dask on {} processes".format(num_processes))
    results = get(dsk, list(dsk.keys()), num_workers=num_processes)
    logging.info("Finished dask")

    tec_mean = np.zeros((D, Nt))
    tec_uncert = np.zeros((D, Nt))
    const_mean = np.zeros((D, Nt))
    const_uncert = np.zeros((D, Nt))

    for c, i in enumerate(range(0, D, D // num_processes)):
        start = i
        stop = min(i + (D // num_processes), D)

        tec_mean[start:stop, :] = results[c][0]
        tec_uncert[start:stop, :] = results[c][1]
        const_mean[start:stop, :] = results[c][2]
        const_uncert[start:stop, :] = results[c][3]

    tec_mean = tec_mean.reshape((Npol, Nd, Na, Nt))
    tec_uncert = tec_uncert.reshape((Npol, Nd, Na, Nt))
    const_mean = const_mean.reshape((Npol, Nd, Na, Nt))
    const_uncert = const_uncert.reshape((Npol, Nd, Na, Nt))

    filter_flags = np.zeros((Npol, Nd, Na, Nt))
    filter_flags = np.tile(filter_flags[:, :, :, None, :], (1, 1, 1, Nf, 1))

    logging.info("Storing results in a datapack")
    datapack.current_solset = 'directionally_referenced'
    # Npol, Nd, Na, Nf, Nt
    datapack.select(**select)
    datapack.tec = tec_mean
    datapack.weights_tec = tec_uncert
    datapack.const = const_mean
    datapack.weights_const = const_uncert
    logging.info("Stored tec and const results. Done")

    tec_conv = -8.4479745e6 / freqs
    phase_model = tec_conv[:, None] * tec_mean[..., None, :] + const_mean[..., None, :]

    Yreal_model = amp_smooth * np.cos(phase_model)
    Yimag_model = amp_smooth * np.sin(phase_model)

    res_real = Yreal_model - Yreal_full.reshape(Yreal_model.shape)
    res_imag = Yimag_model - Yimag_full.reshape(Yimag_model.shape)

    np.savez(os.path.join(output_folder, 'residual_data.npz'), res_real=res_real, res_imag=res_imag)
    print('./residual_data.npz')

    if plot:
        plot_results(Na, Nd, Nt, Yimag_model, Yreal_model, antenna_labels, filter_flags, output_folder, phase_model,
                     phase_raw, res_imag, res_real, tec_mean)


def plot_results(Na, Nd, Nt, Yimag_model, Yreal_model, antenna_labels, filter_flags, output_folder, phase_model,
                 phase_raw, res_imag, res_real, tec_mean):
    os.makedirs(os.path.join(output_folder, 'summary'), exist_ok=True)
    block_size = 960
    for i in range(Na):
        for j in range(Nd):
            for b in range(0, Nt, block_size):

                slice_flag = filter_flags[0, j, i, :, b:b + block_size]

                slice_phase_data = wrap(phase_raw[0, j, i, :, b:b + block_size])
                slice_phase_model = wrap(phase_model[0, j, i, :, b:b + block_size])

                slice_real = Yreal_model[0, j, i, :, b:b + block_size]
                slice_imag = Yimag_model[0, j, i, :, b:b + block_size]
                slice_res_real = res_real[0, j, i, :, b:b + block_size]
                slice_res_imag = res_imag[0, j, i, :, b:b + block_size]
                time_array = np.arange(slice_res_real.shape[-1])
                colors = plt.cm.jet(np.arange(slice_res_real.shape[-1]) / slice_res_real.shape[-1])
                ###
                # empirical covariance
                # Nf, Nt
                _slice_res_real = slice_res_real - np.mean(slice_res_real, axis=0)
                _slice_res_imag = slice_res_imag - np.mean(slice_res_imag, axis=0)
                rr = np.mean(_slice_res_real ** 2, axis=0)
                ii = np.mean(_slice_res_imag ** 2, axis=0)
                ri = np.mean(_slice_res_real * _slice_res_imag, axis=0)
                rho = ri / np.sqrt(rr * ii)

                slice_tec = tec_mean[0, j, i, b:b + block_size]
                fig, axs = plt.subplots(3, 3, figsize=(20, 20))

                diff_phase = wrap(wrap(slice_phase_data) - wrap(slice_phase_model))

                axs[0][2].imshow(slice_phase_data, origin='lower', aspect='auto', cmap='coolwarm', vmin=-np.pi,
                                 vmax=np.pi)
                axs[1][2].imshow(slice_phase_model, origin='lower', aspect='auto', cmap='coolwarm', vmin=-np.pi,
                                 vmax=np.pi)
                axs[2][2].imshow(diff_phase, origin='lower', aspect='auto', cmap='coolwarm', vmin=-0.1, vmax=0.1)

                for nu in range(slice_res_real.shape[-2]):
                    f_c = plt.cm.binary((nu + 1) / slice_res_real.shape[-2])
                    colors_ = (f_c + colors) / 2. * np.array([1., 1., 1., 1. - (nu + 1) / slice_res_real.shape[-2]])
                    colors_[np.where(slice_flag[nu, :])[0], :] = np.array([1., 0., 0., 1.])
                    # print(np.where(slice_flag)[0])

                    axs[0][0].scatter(time_array, np.abs(slice_res_real[nu, :]), c=colors_, marker='.')
                    axs[0][0].scatter(time_array, -np.abs(slice_res_imag[nu, :]), c=colors_, marker='.')
                    axs[2][1].scatter(slice_real[nu, :], slice_imag[nu, :], c=colors_, marker='.')
                axs[0][0].set_title("Real and Imag residuals")
                axs[0][0].hlines(0., time_array.min(), time_array.max())
                axs[2][1].set_title("Real and Imag model")
                axs[2][1].set_xlim(-1.4, 1.4)
                axs[2][1].set_ylim(-1.4, 1.4)
                axs[0][1].scatter(time_array, np.sqrt(rr), c=colors)
                axs[0][1].set_title(r'$\sigma_{\rm real}$')
                axs[1][0].scatter(time_array, np.sqrt(ii), c=colors)
                axs[1][0].set_title(r'$\sigma_{\rm imag}$')
                axs[1][1].scatter(time_array, rho, c=colors)
                axs[1][1].set_title(r'$\rho$')
                axs[2][0].scatter(time_array, slice_tec, c=colors)
                axs[2][0].set_title("TEC")
                plt.tight_layout()

                plt.savefig(os.path.join(output_folder, 'summary',
                                         'fig_{}_{:02d}_{:03d}.png'.format(antenna_labels[i].decode(), j, b)))
                plt.close('all')


if __name__ == '__main__':
    select = dict(ant=slice(50, None, 1), time=slice(0, None, 1), dir=None, freq=None, pol=slice(0, 1, 1))
    distribute_solves('/net/lofar1/data1/albert/imaging/data/lockman/L667218_DDS4_full.h5',
                      ref_dir_idx=0, num_processes=64, select=select, plot=True, debug=True,
                      output_folder='lockman_L667218_tec_vi_run49')
    # update_h5parm('/net/bovenrijn/data1/wwilliams/lba_test/zwcl0634/ddf_kMS_15dir/DDS5_2min_full_merged.h5',
    #               '/home/albert/lofar1_1/imaging/data/LBA/zwcl0634_ddf_kms_15dir_2min_updated.h5')
    # distribute_solves('/home/albert/lofar1_1/imaging/data/LBA/zwcl0634_ddf_kms_15dir_2min_updated.h5',
    #                   ref_dir_idx=0, num_processes=64, select=select, plot=True, debug=True,
    #                   output_folder='LBA_tec_vi_run1')
    # distribute_solves('/net/lofar1/data1/albert/imaging/data/P126+65_compact_raw/P126+65_full_compact_raw_v11.h5',
    #                   ref_dir_idx=0, num_processes=36, solve_type='least_squares', select=select)
    # update_h5parm('/net/lofar1/data1/albert/imaging/data/lockman/L667218_DDS4_full.h5',
    #               '/net/lofar1/data1/albert/imaging/data/lockman/L667218_DDS4_full_updated.h5')
    # old = DataPack('/net/lofar1/data1/albert/imaging/data/lockman/L667218_DDS4_full.h5', readonly=True)
    # old.current_solset = 'sol000'
    # new = DataPack('/net/lofar1/data1/albert/imaging/data/lockman/L667218_DDS4_full_updated.h5', readonly=False)
    # new.current_solset = 'sol000'
    # print(old)
    # print(new)
    # distribute_solves('/home/albert/lofar1_1/imaging/data/LBA/zwcl0634_ddf_kms_15dir_2min.npz',#'/net/bovenrijn/data1/wwilliams/lba_test/zwcl0634/ddf_kMS_15dir/DDS5_2min_full_merged.h5',
    #                   ref_dir_idx=None, num_processes=64, numpy_data=True, elbo_save='/home/albert/lofar1_1/imaging/data/LBA/zwcl0634_ddf_kms_15dir_2min_neg_elbo.npz')
    # distribute_solves('/home/albert/lofar1_1/imaging/data/LBA/zwcl0634_ddf_kms_15dir_1min.npz',#'/net/bovenrijn/data1/wwilliams/lba_test/zwcl0634/ddf_kMS_15dir/DDS5_1min_full_merged.h5',
    #                    ref_dir_idx=None, num_processes=64, numpy_data=True, elbo_save='/home/albert/lofar1_1/imaging/data/LBA/zwcl0634_ddf_kms_15dir_1min_neg_elbo.npz')
