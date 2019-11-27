import os

os.environ['OMP_NUM_THREADS'] = "1"
import numpy as np
import pylab as plt
from scipy.interpolate import griddata
from scipy.optimize import brute, minimize
from scipy.ndimage import median_filter
from bayes_gain_screens import logging
from bayes_gain_screens.datapack import DataPack
from bayes_gain_screens.misc import make_soltab
from bayes_gain_screens.plotting import animate_datapack
from dask.multiprocessing import get
from scipy.optimize import least_squares
import argparse
from timeit import default_timer

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


def log_gaussian_pdf(x, y, sigma):
    # S, Nf
    x = x / sigma
    y = y / sigma
    log_prob = -(x ** 2 + y ** 2) / 2.
    log_prob += -np.log(2 * np.pi) - 2 * np.log(sigma)
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

    def __init__(self, Yreal, Yimag, freqs, laplace_b=None, sigma=None):

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
        log_prob = log_gaussian_pdf((Yreal_m - self.Yreal),
                                    (Yimag_m - self.Yimag),
                                    self.sigma)

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

        log_prob = log_gaussian_pdf((Yreal_m - self.Yreal),
                                    (Yimag_m - self.Yimag),
                                    self.sigma)

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
                "Invalid loss {} var exp {} tec KL {} const KL {} params {} sigma {}".format(loss, var_exp,
                                                                                             tec_prior_KL,
                                                                                             const_prior_KL,
                                                                                             params,
                                                                                             self.sigma))
        # B
        return loss


def sequential_solve(Yreal, Yimag, freqs, working_dir, unravel_index):
    """
    Run on blocks of time.

    :param Yreal:
        [D, Nf, Nt]
    :param Yimag:
    :param freqs:
    :return:
        [D, Nt], [D, Nt]
    """
    plot_period = 600.  # 10 minutes
    loss_dir = os.path.join(working_dir, 'losses')
    os.makedirs(loss_dir, exist_ok=True)
    # debug stuff
    np.random.seed(0)
    tec_range = np.random.uniform(-200, 200, size=15000)
    const_range = np.random.uniform(-np.pi, np.pi, size=15000)
    grid_points = np.stack([x.flatten() for x in
                            np.meshgrid(np.linspace(-200., 200., 401), np.linspace(-np.pi, np.pi, 100), indexing='ij')],
                           axis=1)

    tec_conv = -8.4479745e6 / freqs
    D, Nf, N = Yreal.shape

    tec_mean_array = np.zeros((D, N))
    tec_uncert_array = np.zeros((D, N))
    const_mean_array = np.zeros((D, N))
    const_uncert_array = np.zeros((D, N))

    t0 = -np.inf
    tstart = default_timer()

    count = 0
    for d in range(D):
        dir_idx, ant_idx = unravel_index[0][d], unravel_index[1][d]
        # Nf, Nf N, Nf N
        sigma, Yreal_res, Yimag_res = get_residual_gaussian_statistics(Yimag,
                                                                       Yreal,
                                                                       ant_idx, d, dir_idx)
        phase = np.arctan2(Yimag, Yreal)
        diff_tec = wrap(wrap(phase[d,:,1:]) - wrap(phase[d,:,:-1]))/tec_conv[:,None]

        tec_diff_prior = np.sqrt(np.mean(diff_tec**2))
        const_diff_prior = 0.1
        priors = dict(tec_mean_prior=0., tec_uncert_prior=100., const_mean_prior=0., const_uncert_prior=1.)

        # flag based on smoothed residuals
        total_res = np.sqrt(Yreal_res ** 2 + Yimag_res ** 2)
        weights = total_res < 3. * sigma[:, None]
        with np.printoptions(precision=2):
            logging.info("Average flagging per channel:\n{}".format(1. - weights.mean(1)))
        where_over_flagged = np.where(weights.sum(0) < weights.shape[0] / 2.)[0]
        for t in where_over_flagged:
            logging.info("Time step {} over flagged".format(t))
            weights[:, t] = total_res[:, t] < np.sort(total_res[:, t])[-weights.shape[0] // 2]

        for n in range(N):

            keep = np.where(weights[:, n])[0]

            # least square two pass
            ls_obj = SolveLossMaxLike(Yreal[d, keep, n], Yimag[d, keep, n], freqs[keep], sigma=sigma[keep])
            result1 = list(brute(ls_obj.loss_func,
                                 (slice(-200., 200., 5.),
                                  slice(-np.pi, np.pi, 0.1)),
                                 finish=least_squares))
            result1[1] = np.arctan2(np.sin(result1[1]), np.cos(result1[1]))
            ls_obj = SolveLossVI(Yreal[d, keep, n], Yimag[d, keep, n], freqs[keep], sigma=sigma[keep], **priors)
            result2 = minimize(ls_obj.loss_func,
                               [result1[0], deconstrain(3., 0.01, 55.), result1[1], deconstrain(0.5, 0.001, 2 * np.pi)],
                               method='BFGS', options={'gtol':1e-8, 'norm':2.}).x
            result2[2] = np.arctan2(np.sin(result2[2]), np.cos(result2[2]))
            tec_mean, _tec_uncert, const_mean, _const_uncert = result2
            tec_uncert = constrain(_tec_uncert, 0.01, 55.)
            const_uncert = constrain(_const_uncert, 0.001, 2 * np.pi)
            tec_mean_array[d, n] = tec_mean
            tec_uncert_array[d, n] = tec_uncert
            const_mean_array[d, n] = const_mean
            const_uncert_array[d, n] = const_uncert

            if n > 50:
                # dt0 = np.diff(tec_mean_array[d, :n + 1])
                dc0 = np.diff(const_mean_array[d, :n + 1])
                dc = dc0
                for _ in range(3):
                    # dt = np.where(np.abs(dt0) > 3 * np.nanstd(dt), np.nan, dt0)
                    dc = np.where(np.abs(dc0) > 3 * np.nanstd(dc), np.nan, dc0)
                # tec_diff_prior = 1.5 * max(1., np.sqrt(np.nanmean(dt ** 2)))
                const_diff_prior = 1.5 * max(0.01, np.sqrt(np.nanmean(dc ** 2)))

            priors['tec_mean_prior'] = tec_mean
            priors['tec_uncert_prior'] = np.sqrt(tec_diff_prior ** 2 + tec_uncert ** 2)
            priors['const_mean_prior'] = const_mean
            priors['const_uncert_prior'] = np.sqrt(const_diff_prior ** 2 + const_uncert ** 2)
            # logging.info("dir {:02d} ant {:02d} time {:03d}:\n\ttec {:.2f} +- {:.2f} const {:.2f} +- {:.2f} [diff {:.2f} {:.2f}]".format(dir_idx, ant_idx, n,
            #                                                                                    tec_mean,
            #                                                                                    tec_uncert, const_mean,
            #                                                                                    const_uncert,
            #                                                                                     tec_mean - tec_mean_array[d,n-1],
            #                                                                                     const_mean - const_mean_array[d,n-1]))
            count += 1
            if default_timer() - t0 > plot_period:
                logging.info(
                    "Time: {:.2f} minutes ({:.2f}%) - dir {:02d} ant {:02d}".format((default_timer() - tstart) / 60.,
                                                                                    count / (N * D) * 100., dir_idx,
                                                                                    ant_idx))
                t0 = default_timer()
                ls_obj = SolveLossMaxLike(Yreal[d, :, n], Yimag[d, :, n], freqs, sigma=sigma)
                loss = np.array([ls_obj.loss_func([t, c]) for t, c in zip(tec_range, const_range)])
                grid_loss = griddata((tec_range, const_range), loss, grid_points, method='linear').reshape((401, 100)).T

                fig, axs = plt.subplots(2, 1, figsize=(6, 6))
                axs[0].imshow(grid_loss, origin='lower', aspect='auto',
                              extent=(grid_points[:, 0].min(), grid_points[:, 0].max(),
                                      grid_points[:, 1].min(), grid_points[:, 1].max()))
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
                axs[1].legend(frameon=False)
                axs[1].scatter(freqs, ls_obj.phase, label='data')
                axs[1].set_xlabel('freq (Hz)')
                axs[1].set_ylabel('phase (rad)')
                plt.savefig(
                    os.path.join(loss_dir, 'loss_dir{:02d}_ant{:02d}_time{:03d}.png'.format(dir_idx, ant_idx, n)))
                plt.close('all')

    return tec_mean_array, tec_uncert_array, const_mean_array, const_uncert_array


def get_residual_gaussian_statistics(Yimag, Yreal, ant_idx, d, dir_idx):
    _, Nf, N = Yimag.shape
    _times = np.linspace(-1., 1., N)
    _freqs = np.linspace(-1., 1., Nf)

    Yreal = Yreal[d, :, :]
    Yimag = Yimag[d, :, :]

    Yreal_smooth, Yimag_smooth = smooth_gains(Yreal, Yimag, filter_size=3, deg=2)

    Yreal_res = Yreal - Yreal_smooth
    Yimag_res = Yimag - Yimag_smooth

    sigma = np.sqrt(np.mean(Yreal_res ** 2 + Yimag_res ** 2, axis=-1))
    with np.printoptions(precision=2):
        logging.info("For dir {:02d} ant {:02d}: Found Gaussian sigma:\n\t{}".format(dir_idx, ant_idx, sigma))
    return sigma, Yreal_res, Yimag_res


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


def main(data_dir, working_dir, obs_num, ref_dir, ncpu):
    os.chdir(working_dir)
    logging.info("Performing TEC and constant variational inference.")
    merged_h5parm = os.path.join(data_dir, 'L{}_{}_merged.h5'.format(obs_num, 'DDS4_full'))
    select = dict(pol=slice(0, 1, 1))
    datapack = DataPack(merged_h5parm, readonly=False)
    logging.info("Creating directionally_referenced/tec000+const000")
    make_soltab(datapack, from_solset='sol000', to_solset='directionally_referenced', from_soltab='phase000',
                to_soltab=['tec000', 'const000'])
    logging.info("Getting raw phase")
    datapack.current_solset = 'sol000'
    datapack.select(**select)
    axes = datapack.axes_phase
    antenna_labels, antennas = datapack.get_antennas(axes['ant'])
    patch_names, directions = datapack.get_directions(axes['dir'])
    timestamps, times = datapack.get_times(axes['time'])
    freq_labels, freqs = datapack.get_freqs(axes['freq'])
    pol_labels, pols = datapack.get_pols(axes['pol'])
    Npol, Nd, Na, Nf, Nt = len(pols), len(directions), len(antennas), len(freqs), len(times)
    phase_raw, axes = datapack.phase
    logging.info("Getting smooth phase and amplitude data")
    datapack.current_solset = 'smoothed000'
    datapack.select(**select)
    phase_smooth, axes = datapack.phase
    amp_smooth, axes = datapack.amplitude
    logging.info("Directionally referencing phase.")
    logging.info("Using ref_dir {}".format(ref_dir))
    phase_di = phase_smooth[:, ref_dir:ref_dir + 1, ...]
    phase_dd = phase_raw - phase_di

    # Npol, Nd, Na, Nf, Nt
    Yimag = amp_smooth * np.sin(phase_dd)
    Yreal = amp_smooth * np.cos(phase_dd)

    Yimag = Yimag.reshape((-1, Nf, Nt))
    Yreal = Yreal.reshape((-1, Nf, Nt))

    logging.info("Creating dask.")

    def d_map(start, stop):
        return np.unravel_index(range(start, stop), (Nd, Na))

    D = Yimag.shape[0]
    num_processes = min(D, ncpu)
    dsk = {}
    for i in range(0, D, D // num_processes):
        start = i
        stop = min(i + (D // num_processes), D)
        dsk[str(i)] = (sequential_solve, Yreal[start:stop, :, :], Yimag[start:stop, :, :], freqs, working_dir,
                       d_map(start, stop))
    logging.info("Running dask on {} processes".format(num_processes))
    results = get(dsk, list(dsk.keys()), num_workers=num_processes)
    logging.info("Finished dask.")
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

    logging.info("Finding outliers based on mean residual")
    tec_conv = -8.4479745e6 / freqs
    phase_model = tec_conv[:, None] * tec_mean[..., None, :] + const_mean[..., None, :]

    Yreal_model = amp_smooth * np.cos(phase_model)
    Yimag_model = amp_smooth * np.sin(phase_model)

    res_real = Yreal_model - Yreal.reshape(Yreal_model.shape)
    res_imag = Yimag_model - Yimag.reshape(Yimag_model.shape)

    # total_res0 = np.sqrt(res_real ** 2 + res_imag ** 2)
    # total_res = total_res0
    # print(total_res.shape)
    # for _ in range(3):
    #     m = total_res0 > 3. * np.nanstd(total_res, axis=-1, keepdims=True)
    #     print(m.shape)
    #     total_res = np.where(m, np.nan, total_res0)
    #     print(total_res.shape)
    # flag = np.nanmean(total_res0, axis=-2) > 3. * np.nanstd(
    #     np.nanmean(total_res, axis=-2), axis=-1, keepdims=True)
    # tec_uncert[flag] = np.inf

    logging.info("Storing results")
    datapack.current_solset = 'directionally_referenced'
    # Npol, Nd, Na, Nf, Nt
    datapack.select(**select)
    datapack.tec = tec_mean
    datapack.weights_tec = tec_uncert
    datapack.const = const_mean
    datapack.weights_const = const_uncert
    logging.info("Stored tec and const results. Done")

    # animate_datapack(merged_h5parm, os.path.join(working_dir, 'tec_plots_flagged'), num_processes=ncpu,
    #                  solset='directionally_referenced',
    #                  observable='tec', vmin=-60., vmax=60., plot_facet_idx=True,
    #                  labels_in_radec=True, plot_crosses=False, phase_wrap=False,
    #                  flag_outliers=True)

    animate_datapack(merged_h5parm, os.path.join(working_dir, 'tec_plots'), num_processes=ncpu,
                     solset='directionally_referenced',
                     observable='tec', vmin=-60., vmax=60., plot_facet_idx=True,
                     labels_in_radec=True, plot_crosses=False, phase_wrap=False,
                     flag_outliers=False)

    animate_datapack(merged_h5parm, os.path.join(working_dir, 'const_plots'), num_processes=ncpu,
                     solset='directionally_referenced',
                     observable='const', vmin=-np.pi, vmax=np.pi, plot_facet_idx=True,
                     labels_in_radec=True, plot_crosses=False, phase_wrap=True,
                     flag_outliers=False)

    plot_results(Na, Nd, antenna_labels, working_dir, phase_model, phase_dd,
                 res_imag, res_real, tec_mean, const_mean)


def plot_results(Na, Nd, antenna_labels, working_dir, phase_model,
                 phase_dd, res_imag, res_real, tec_mean, const_mean):
    logging.info("Plotting results.")
    summary_dir = os.path.join(working_dir, 'summaries')
    os.makedirs(summary_dir, exist_ok=True)
    for i in range(Na):
        for j in range(Nd):
            slice_phase_data = wrap(phase_dd[0, j, i, :, :])
            slice_phase_model = wrap(phase_model[0, j, i, :, :])
            slice_res_real = res_real[0, j, i, :, :]
            slice_res_imag = res_imag[0, j, i, :, :]
            time_array = np.arange(slice_res_real.shape[-1])
            colors = plt.cm.jet(np.arange(slice_res_real.shape[-1]) / slice_res_real.shape[-1])
            # Nf, Nt
            _slice_res_real = slice_res_real - np.mean(slice_res_real, axis=0)
            _slice_res_imag = slice_res_imag - np.mean(slice_res_imag, axis=0)
            slice_tec = tec_mean[0, j, i, :]
            slice_const = const_mean[0, j, i, :]
            fig, axs = plt.subplots(2, 2, figsize=(15, 15))
            diff_phase = wrap(wrap(slice_phase_data) - wrap(slice_phase_model))
            for nu in range(slice_res_real.shape[-2]):
                f_c = plt.cm.binary((nu + 1) / slice_res_real.shape[-2])
                colors_ = (f_c + colors) / 2. * np.array([1., 1., 1., 1. - (nu + 1) / slice_res_real.shape[-2]])
                axs[0][0].scatter(time_array, np.abs(slice_res_real[nu, :]), c=colors_, marker='.')
                axs[0][0].scatter(time_array, -np.abs(slice_res_imag[nu, :]), c=colors_, marker='.')
            axs[0][0].set_title("Real and Imag residuals")
            axs[0][0].hlines(0., time_array.min(), time_array.max())
            axs[0][1].imshow(diff_phase, origin='lower', aspect='auto', cmap='coolwarm', vmin=-0.2,
                             vmax=0.2)
            axs[0][1].set_title('Phase residuals [-0.2,0.2]')
            axs[0][1].set_xlabel('Time')
            axs[0][1].set_ylabel('Freq')
            axs[1][0].scatter(time_array, slice_tec, c=colors)
            axs[1][0].set_title("TEC")
            axs[1][0].set_xlabel('Time')
            axs[1][0].set_ylabel('TEC [mTECU]')
            axs[1][1].scatter(time_array, slice_const, c=colors)
            axs[1][1].set_title("Constant")
            axs[1][1].set_xlabel('Time')
            axs[1][1].set_ylabel('Const [radians]')
            plt.tight_layout()
            plt.savefig(os.path.join(summary_dir, 'summary_{}_dir{:02d}.png'.format(antenna_labels[i].decode(), j)))
            plt.close('all')


def add_args(parser):
    parser.add_argument('--obs_num', help='Obs number L*',
                        default=None, type=int, required=True)
    parser.add_argument('--data_dir', help='Where are the ms files are stored.',
                        default=None, type=str, required=True)
    parser.add_argument('--working_dir', help='Where to perform the imaging.',
                        default=None, type=str, required=True)
    parser.add_argument('--ncpu', help='How many processors available.',
                        default=None, type=int, required=True)
    parser.add_argument('--ref_dir', help='The index of reference dir.',
                        default=0, type=int, required=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Smoothes the DDS4_full solutions and stores in smoothed000 solset.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_args(parser)
    flags, unparsed = parser.parse_known_args()
    print("Running with:")
    for option, value in vars(flags).items():
        print("    {} -> {}".format(option, value))
    main(**vars(flags))
