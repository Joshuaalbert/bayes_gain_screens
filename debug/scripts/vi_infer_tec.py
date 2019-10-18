import os

os.environ['OMP_NUM_THREADS'] = "1"
import numpy as np
import pylab as plt
from matplotlib.mlab import griddata
from scipy.optimize import brute, fmin
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
    # if ~np.all(np.isfinite(log_prob)):
    #     raise ValueError("Invalid log_prob {} x {} y {} laplace {} {} {}".format(log_prob, x, y, b1, b2, rho))
    log_prob += -np.log(np.pi * b1 * b2 * np.sqrt(1. - rho ** 2))
    # if ~np.all(np.isfinite(log_prob)):
    #     raise ValueError("Invalid log_prob {} laplace {} {} {}".format(log_prob, b1, b2, rho))

    return log_prob


def log_laplace_uncorr_pdf(x, y, b):
    x = x / b
    y = y / b
    log_prob = np.log(k0(np.sqrt(2. * (x ** 2 + y ** 2))))
    log_prob += -np.log(np.pi * b ** 2)
    return log_prob


class SolveLoss(object):
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
                 S=40, laplace_b=None):
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
        self.loss_func = self._tec_only_loss_func_corr
        self.laplace_b = laplace_b

    def calculate_residuals(self, params):
        phase = self.phase_model(params)
        Yreal_m = self.amp * np.cos(phase)
        Yimag_m = self.amp * np.sin(phase)
        res_real = (self.Yreal - Yreal_m)
        res_imag = (self.Yimag - Yimag_m)
        return res_real, res_imag

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

    def _tec_only_loss_func_corr(self, params):
        """
        VI loss
        :param params: tf.Tensor
            shapfrom scipy.optimize import brute, fmine [D]
        :return: tf.Tensor
            scalar The loss
        """
        tec_mean, log_tec_uncert, const_mean, log_const_uncert = params[0], params[1], params[2]

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
        inf_mask = np.isinf(log_prob)
        #Nf
        # scale = inf_mask.size/(inf_mask.size - inf_mask.sum(0).sum(0))
        log_prob = np.where(inf_mask, 0., log_prob)

        # S_tec, S_const
        log_prob = np.sum(log_prob, axis=-1)
        # scalar
        var_exp = np.sum(np.sum(log_prob * self.w_tec[:, None], axis=0) * self.w_const, axis=0)

        # Get KL
        tec_prior_KL = self._scalar_KL(tec_mean, tec_uncert, self.tec_mean_prior, self.tec_uncert_prior)
        const_prior_KL = self._scalar_KL(const_mean, const_uncert, self.const_mean_prior, self.const_uncert_prior)
        loss = np.negative(var_exp - tec_prior_KL - const_prior_KL)
        if ~np.isfinite(loss):
            raise ValueError("Invalid loss {} var exp {} tec KL {} const KL {} params {} laplace {}".format(loss, var_exp, tec_prior_KL, const_prior_KL, params, self.laplace_b))
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
        debug_log_uncert_range = np.random.uniform(np.log(0.1), np.log(50.), size=15000)
        debug_grid_points = np.linspace(-200., 200., 401), np.linspace(np.log(0.1), np.log(50.), 100)
        if not callable(debug_indicator) and isinstance(debug_indicator, bool):
            debug_indicator = lambda *x: debug_indicator

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    priors = dict(tec_mean_prior=0., tec_uncert_prior=100.)

    D, Nf, N = Yreal.shape

    Yreal_res = np.zeros_like(Yreal)
    Yimag_res = np.zeros_like(Yimag)
    phase_res = np.zeros_like(Yimag)

    tec_mean_array = np.zeros((D, N))
    tec_uncert_array = np.zeros((D, N))
    const_array = np.zeros((D, N))

    for d in range(D):
        dir_idx, ant_idx = unravel_index[0][d], unravel_index[1][d]
        # 3, Nf
        laplace_b = get_residual_laplace_statistics(Yimag,
                                                    Yreal,
                                                    ant_idx, d, dir_idx, output_dir)
        laplace_b[0,:] = np.maximum(laplace_b[0,:], 0.001)
        laplace_b[1,:] = np.maximum(laplace_b[1,:], 0.001)
        tec_diff_prior = 50.

        Yreal_smooth, Yimag_smooth = smooth_gains(Yreal[d, :, :], Yimag[d, :, :], time_kernel_size=5)

        Yreal_res = Yreal[d, :, :] - Yreal_smooth
        Yimag_res = Yimag[d, :, :] - Yimag_smooth
        total_res = np.sqrt(Yreal_res**2 + Yimag_res**2)
        weights = total_res > np.percentile(total_res, 95)

        for n in range(N):

            keep = np.where(weights[:, n])[0]

            # least square two pass
            ls_obj = SolveLoss(Yreal[d, keep, n], Yimag[d, keep, n], freqs[keep], laplace_b=laplace_b[:, keep],
                               **priors)
            result = brute(ls_obj.loss_func,
                           (slice(-200., 200., 5.),
                            slice(np.log(0.01), np.log(10.), 1.),
                            slice(-np.pi, np.pi, 0.05)),
                           finish=least_squares)

            tec_mean, log_tec_uncert, const = result
            tec_uncert = 0.1 + np.exp(log_tec_uncert)
            tec_mean_array[d, n] = tec_mean
            tec_uncert_array[d, n] = tec_uncert
            const_array[d, n] = const

            # if n > 20:
            #     tec_diff_prior = np.maximum(np.sqrt(np.mean(np.square(np.diff(tec_mean_array[d, :n+1])))), 10.)
            priors['tec_mean_prior'] = tec_mean
            priors['tec_uncert_prior'] = np.sqrt(tec_diff_prior ** 2 + tec_uncert ** 2)

            logging.info(
                "dir {:02d} ant {:02d} time {:03d} tec {} +- {} const {}".format(dir_idx, ant_idx, n, tec_mean, tec_uncert, const))

            if debug_indicator is not None:
                if debug_indicator(d, n):
                    logging.info("Plotting {} {}".format(d, n))
                    debug_elbo = np.array([ls_obj.loss_func([t, ltu, const]) for t, ltu in
                                           zip(debug_tec_range, debug_log_uncert_range)])
                    debug_grid_elbo = griddata(debug_tec_range, debug_log_uncert_range, debug_elbo, *debug_grid_points,
                                               interp='linear')

                    fig, axs = plt.subplots(2, 1, figsize=(6, 6))
                    img = axs[0].imshow(debug_grid_elbo, origin='lower', aspect='auto',
                                        extent=(debug_grid_points[0].min(), debug_grid_points[0].max(),
                                                debug_grid_points[1].min(), debug_grid_points[1].max()))
                    # plt.colorbar(img)
                    axs[0].set_ylabel('log (tec uncertainty[mTECU])')
                    axs[0].set_xlabel('tec [mTECU]')
                    xlim = axs[0].get_xlim()
                    ylim = axs[0].get_ylim()
                    axs[0].scatter(np.clip(result[0], *xlim), np.clip(result[1], *ylim), c='red')
                    axs[0].set_xlim(*xlim)
                    axs[0].set_ylim(*ylim)
                    axs[0].set_title("Solved {:.1f} +- {:.1f} mTECU const {:.2f} rad".format(result[0], np.exp(result[1]), const))
                    axs[1].plot(freqs, wrap(ls_obj.phase_model(result)), c='red', label='mean phase')
                    axs[1].plot(freqs, wrap(ls_obj.phase_model([result[0] + result[1], None, const])), c='black', ls='dotted',
                                label=r'$+\sigma$ phase')
                    axs[1].plot(freqs, wrap(ls_obj.phase_model([result[0] - result[1], None, const])), c='black', ls='dashed',
                                label=r'$-\sigma$ phase')
                    axs[1].scatter(freqs, ls_obj.phase, label='data')
                    axs[1].set_xlabel('freq (Hz)')
                    axs[1].set_ylabel('phase (rad)')
                    os.makedirs(os.path.join(output_dir, 'elbo'), exist_ok=True)
                    plt.savefig(os.path.join(output_dir, 'elbo',
                                             'elbo_dir{:02d}_ant{:02d}_time{:03d}.png'.format(dir_idx, ant_idx, n)))
                    plt.close('all')

    return tec_mean_array, tec_uncert_array, const_array


def get_residual_laplace_statistics(Yimag, Yreal, ant_idx, d, dir_idx, output_dir):
    logging.info("Using Rbf to get noise levels for dir {:02d} ant {:02d}.".format(dir_idx, ant_idx))

    hist_dir = os.path.join(output_dir, 'histograms')
    os.makedirs(hist_dir, exist_ok=True)
    _, Nf, N = Yimag.shape
    _times = np.linspace(-1., 1., N)
    _freqs = np.linspace(-1., 1., Nf)

    Yreal = Yreal[d, :, :]
    Yimag = Yimag[d, :, :]

    Yreal_smooth, Yimag_smooth = smooth_gains(Yreal, Yimag, time_kernel_size=5)

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
            rho = 0.99*np.tanh(arctanh_rho)
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
        rho = 0.99*np.tanh(arctanh_rho)
        # b1, b2, rho = [0.1, 0.1, 0.]

        logging.info(
            "freq {:02d} dir {:02d} ant {:02d}: Found laplace scale {:.3f} {:.3f} {:.3f}".format(nu, dir_idx, ant_idx, b1, b2, rho))
        laplace_b_freq.append([b1, b2, rho])

        f, axs = plt.subplots(2, 1, figsize=(4, 8))
        axs[0].imshow(H.T, extent=(binsxc.min(), binsxc.max(),
                                   binsyc.min(), binsyc.max()), vmin=0., vmax=H.max(), origin='lower',
                      aspect='auto')

        axs[1].imshow(np.exp(_laplace_logpdf(log_b1, log_b2, arctanh_rho).T), extent=(binsxc.min(), binsxc.max(),
                                                                                      binsyc.min(), binsyc.max()),
                      vmin=0.,
                      vmax=H.max(), origin='lower', aspect='auto')
        axs[0].set_title("b1 {:.2f} b2 {:.2f} rho {:.2f}".format(b1,b2,rho))
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

def smooth_gains(Yreal, Yimag, time_kernel_size=5):
    """

    :param Yreal: [Nf, N]
    :param Yimag: [Nf, N]
    :param size:
    :return: Yreal, Yimag
        [Nf, N], [Nf, N]
    """
    Nf, N = Yreal.shape
    _freqs = np.linspace(-1., 1., Nf)
    real = sum([median_filter(p, time_kernel_size)*_freqs[:,None]**(1-i) for i, p in enumerate(np.polyfit(_freqs,Yreal,deg=1))])
    imag = sum([median_filter(p, time_kernel_size)*_freqs[:,None]**(1-i) for i, p in enumerate(np.polyfit(_freqs,Yimag,deg=1))])
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
    logging.info("Directionally referencing phase.")
    if ref_dir_idx is None:
        ref_dir_idx = np.argmin(np.median(
            np.reshape(np.transpose(np.abs(np.diff(phase_raw, axis=-2)), (1, 0, 2, 3, 4)), (Nd, -1)), axis=1))
    logging.info("Using ref_dir_idx {}".format(ref_dir_idx))
    phase_di = phase_raw[:, ref_dir_idx:ref_dir_idx + 1, ...]
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
    const = np.zeros((D, Nt))

    for c, i in enumerate(range(0, D, D // num_processes)):
        start = i
        stop = min(i + (D // num_processes), D)

        tec_mean[start:stop, :] = results[c][0]
        tec_uncert[start:stop, :] = results[c][1]
        const[start:stop, :] = results[c][2]

    tec_mean = tec_mean.reshape((Npol, Nd, Na, Nt))
    tec_uncert = tec_uncert.reshape((Npol, Nd, Na, Nt))
    const = const.reshape((Npol, Nd, Na, Nt))

    filter_flags = np.zeros((Npol, Nd, Na, Nt))
    filter_flags = np.tile(filter_flags[:, :, :, None, :], (1, 1, 1, Nf, 1))

    logging.info("Storing results in a datapack")
    datapack.current_solset = 'directionally_referenced'
    # Npol, Nd, Na, Nf, Nt
    datapack.select(**select)
    datapack.tec = tec_mean
    datapack.weights_tec = tec_uncert
    datapack.const = const
    logging.info("Stored tec and const results. Done")

    tec_conv = -8.4479745e6 / freqs
    phase_model = tec_conv[:, None] * tec_mean[..., None, :] + const[..., None, :]

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
                    colors_[np.where(slice_flag[nu,:])[0],:] = np.array([1., 0., 0., 1.])
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


def test_numpy_data():
    Npol, Nd, Na, Nf, Nt = 1, 3, 1, 24, 10
    tec_true = np.random.uniform(-200, 200, size=[Npol, Nd, Na, 1]) + np.random.normal(0., 20., size=[1, 1, 1, Nt])
    freqs = np.linspace(100., 160., Nf) * 1e6
    tec_conv = -8.4479745e6 / freqs
    clock_true = 0 * np.random.normal(0., 0.4, size=[Npol, 1, Na, Nt])
    const_true = 0 * np.random.normal(0., 0.5, size=[Npol, 1, Na, Nt])
    phase_true = tec_true[..., None, :] * tec_conv[:, None] + (2. * np.pi * freqs[:, None]) * clock_true[..., None,
                                                                                              :] * 1e-9 + const_true[
                                                                                                          ..., None,
                                                                                                          :]
    amp_true = np.random.uniform(.3, 1.5, size=[Npol, Nd, Na, 1,
                                                Nt])  # + np.random.uniform(-0.1, 0.1, size=[Npol, Nd, Na, Nf, Nt])
    Yreal = amp_true * np.cos(phase_true) + np.random.laplace(scale=0.1, size=phase_true.shape)
    Yimag = amp_true * np.sin(phase_true) + np.random.laplace(scale=0.1, size=phase_true.shape)

    phase_raw = np.arctan2(Yimag, Yreal)
    amp_raw = np.sqrt(Yreal ** 2 + Yimag ** 2)

    return phase_raw, amp_raw, freqs


def transform_old_h5parm(filename, output_file):
    import tables as tb
    with tb.open_file(filename, mode='r') as f:
        print("Axes order: {}".format(f.root.sol000.phase000.val.attrs['AXES']))
        phase = f.root.sol000.phase000.val[...].transpose((4, 3, 2, 1, 0))
        amplitude = f.root.sol000.amplitude000.val[...].transpose((4, 3, 2, 1, 0))
        freqs = f.root.sol000.phase000.freq[...]
        np.savez(output_file, phase=phase, amplitude=amplitude, freqs=freqs)


if __name__ == '__main__':
    select = dict(ant=slice(50, None, 1), time=slice(0, None, 1), dir=None, freq=None, pol=slice(0, 1, 1))
    distribute_solves('/net/lofar1/data1/albert/imaging/data/lockman/L667218_DDS4_full.h5',
                      ref_dir_idx=0, num_processes=64, select=select, plot=True, debug=True,
                      output_folder='lockman_L667218_tec_vi_run39')
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
