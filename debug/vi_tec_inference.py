import numpy as np
import pylab as plt
from matplotlib.mlab import griddata
import os
from scipy.optimize import brute, fmin
from scipy.ndimage import median_filter
from scipy.stats import circstd
from scipy.linalg import solve_triangular
from bayes_gain_screens import logging
from bayes_gain_screens.datapack import DataPack, update_h5parm
from bayes_gain_screens.misc import make_soltab
from dask.multiprocessing import get
from collections import deque
from scipy.optimize import least_squares
from scipy.interpolate import interp1d


"""
This script is still being debugged/tested. 
Get's TEC from gains.
"""


def wrap(p):
    return np.arctan2(np.sin(p), np.cos(p))

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

    def __init__(self, Yreal, Yimag, freqs, scale_real=None, scale_imag=None, tec_mean_prior=0., tec_uncert_prior=100.,
                 S=20, L=None):
        self.x, self.w = np.polynomial.hermite.hermgauss(S)
        self.w /= np.pi

        self.tec_conv = -8.4479745e6 / freqs
        # Nf
        self.amp = np.sqrt(np.square(Yreal) + np.square(Yimag))
        self.Yreal = Yreal
        self.Yimag = Yimag


        self.phase = wrap(np.arctan2(Yimag, Yreal))

        self.scale_real = scale_real
        self.scale_imag = scale_imag

        self.tec_mean_prior = tec_mean_prior
        self.tec_uncert_prior = tec_uncert_prior

        self.phase_model = lambda params: self.tec_conv * params[0]
        self.loss_func = self._tec_only_loss_func
        self.L = L

    def calculate_residuals(self, params):
        phase = self.phase_model(params)
        Yreal_m = self.amp * np.cos(phase)
        Yimag_m = self.amp * np.sin(phase)
        res_real = np.abs(self.Yreal - Yreal_m)
        res_imag = np.abs(self.Yimag - Yimag_m)
        return res_real, res_imag

    def calculate_dist_params(self, params):
        res_real, res_imag = self.calculate_residuals(params)
        scale_real = np.median(res_real)
        scale_imag = np.median(res_imag)
        return scale_real, scale_imag

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

    def _scalar_KL_singular(self, mean, mean_prior, uncert_prior):
        # Get KL
        log_norm = -0.5*np.log(2*np.pi) - np.log(uncert_prior) - 0.5*(mean-mean_prior)**2/uncert_prior**2
        return -log_norm

    def _least_sqaures_loss_func_phase(self, params, weights=None):
        """
        least squares loss

        :param params: [3]
        :param weights: [Nf]
        :return: [Nf]
        """
        #  Nf
        phase = self.phase_model(params)
        res = wrap(wrap(phase) - self.phase)

        if weights is not None:
            return res*weights
        else:
            return res

    def _tec_only_loss_func(self, params, weights):
        """
        VI loss
        :param params: tf.Tensor
            shapfrom scipy.optimize import brute, fmine [D]
        :return: tf.Tensor
            scalar The loss
        """
        tec_mean, log_tec_uncert = params[0], params[1]

        tec_uncert = np.exp(log_tec_uncert)

        weights = weights * weights.size / np.sum(weights)

        # S
        tec = tec_mean + np.sqrt(2.) * tec_uncert * self.x
        # S, Nf
        phase = tec[:, None] * self.tec_conv
        Yreal_m = self.amp * np.cos(phase)
        Yimag_m = self.amp * np.sin(phase)

        # 2, S*Nf
        res = np.reshape(np.stack([Yreal_m - self.Yreal, Yimag_m - self.Yimag], axis=0), (2, self.tec_conv.size*self.w.size))
        # 2, S, Nf
        A = np.reshape(solve_triangular(self.L, res), (2, self.w.size, self.tec_conv.size))
        # S
        maha = 0.5*np.mean(np.sum(np.square(A), axis=0)*weights,axis=1)
        # 2
        log_prob = -np.log(2.*np.pi) - np.log(self.L[0,0]) - np.log(self.L[1,1]) - maha
        var_exp = np.sum(log_prob * self.w)

        # Get KL
        tec_prior_KL = self._scalar_KL(tec_mean, tec_uncert, self.tec_mean_prior, self.tec_uncert_prior)
        loss = np.negative(var_exp - tec_prior_KL)
        # B
        return loss


def post_mean(X, Y, sigma_y, amp, l, X2):
    def _kern(x1, x2):
        dx = -0.5 * np.square((x1[:, None] - x2[None, :]) / l)
        return np.exp(2*np.log(amp) + dx)

    K = _kern(X, X)
    Ks = _kern(X, X2)
    L = np.linalg.cholesky(K + sigma_y ** 2 * np.eye(X.shape[0]))
    A = solve_triangular(L, Ks, lower=True)
    return A.T.dot(solve_triangular(L, Y[:, None], lower=True))[:, 0]

def sequential_solve(Yreal, Yimag, freqs, tec_init, debug_indicator=None,output_dir=None):
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
        #debug stuff
        np.random.seed(0)
        debug_tec_range = np.random.uniform(-200, 200, size=15000)
        debug_log_uncert_range = np.random.uniform(np.log(0.1), np.log(50.), size=15000)
        debug_grid_points = np.linspace(-200., 200., 401), np.linspace(np.log(0.1), np.log(50.), 100)
        if not callable(debug_indicator) and isinstance(debug_indicator, bool):
            debug_indicator = lambda *x: debug_indicator

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)


    priors = dict(tec_mean_prior=0., tec_uncert_prior=100., L=np.eye(2)*0.1)

    D, Nf, N = Yreal.shape
    
    Yreal_res = np.zeros_like(Yreal)
    Yimag_res = np.zeros_like(Yimag)
    phase_res = np.zeros_like(Yimag)


    tec_mean_array = np.zeros((D, N))
    tec_uncert_array = np.zeros((D, N))
    flag_array = np.zeros((D, N))
    weights = np.ones((Nf,))

    for d in range(D):

        tec_diff_prior = 40.

        for n in range(N):

            # least square two pass
            ls_obj = SolveLoss(Yreal[d, :, n], Yimag[d, :, n], freqs, **priors)
            result = brute(ls_obj.loss_func, (slice(-150., 150., 5.), slice(np.log(0.5), np.log(10.), 1.)),
                           args=(weights,))

            #Nf
            res_real0, res_imag0 = ls_obj.calculate_residuals(result)

            curr_phase_res = wrap(wrap(ls_obj.phase_model(result)) - ls_obj.phase)

            if n > 10:
                # print(curr_phase_res, np.nanmedian(np.abs(phase_res[d, :, :n])))
                # print(res_real0,  np.nanmedian(np.abs(Yreal_res[d, :, :n])))
                # print(res_imag0,  np.nanmedian(np.abs(Yimag_res[d, :, :n])))
                phase_flag = np.abs(curr_phase_res) > 4.*np.nanmedian(np.abs(phase_res[d, :, :n]))
                real_flag = np.abs(res_real0) > 4.*np.nanmedian(np.abs(Yreal_res[d, :, :n]))
                imag_flag = np.abs(res_imag0) > 4.*np.nanmedian(np.abs(Yimag_res[d, :, :n]))

                flag = np.logical_or(phase_flag, real_flag, imag_flag)
                weights = np.logical_not(flag)

                if weights.sum() < freqs.size//2:
                    weights = np.abs(res_real0) + np.abs(res_imag0) < np.median(np.abs(res_real0) + np.abs(res_imag0))
                if weights.sum() < freqs.size // 2:
                    weights = np.ones(Nf)
            else:
                weights = np.ones(Nf)
            not_weights = np.logical_not(weights)
            keep = np.where(weights)[0]
            res_real0 = res_real0[keep]
            res_imag0 = res_imag0[keep]
            res_real0, res_imag0 = res_real0 - np.mean(res_real0), res_imag0 - np.mean(res_imag0)
            rr = np.mean(res_real0 ** 2)
            ii = np.mean(res_imag0 ** 2)
            ri = np.mean(res_real0 * res_imag0)
            cov = np.array([[rr, ri] , [ri, ii]])
            try:
                priors['L'] = np.linalg.cholesky(cov)
            except:
                pass

            # round two
            ls_obj = SolveLoss(Yreal[d, keep, n], Yimag[d, keep, n], freqs[keep], **priors)

            result = brute(ls_obj.loss_func, (slice(-150., 150., 5.), slice(np.log(0.5), np.log(10.), 1.)),
                           args=(weights[keep],))

            ls_obj = SolveLoss(Yreal[d, :, n], Yimag[d, :, n], freqs, **priors)
            # Nf
            res_real1, res_imag1 = ls_obj.calculate_residuals(result)
            res_real1[not_weights] = np.nan
            res_imag1[not_weights] = np.nan

            curr_phase_res = wrap(wrap(ls_obj.phase_model(result)) - ls_obj.phase)
            curr_phase_res[not_weights] = np.nan

            Yreal_res[d, :, n] = res_real1
            Yimag_res[d, :, n] = res_imag1
            phase_res[d, :, n] = curr_phase_res

            tec_mean, log_tec_uncert = result
            tec_uncert = np.exp(log_tec_uncert)
            tec_mean_array[d, n] = tec_mean
            tec_uncert_array[d, n] = tec_uncert

            if n > 10:
                tec_diff_prior = np.maximum(np.sqrt(np.mean(np.square(np.diff(tec_mean_array[d, :n+1])))), 5.)
            priors['tec_mean_prior'] = tec_mean
            priors['tec_uncert_prior'] = np.sqrt(tec_diff_prior**2 + tec_uncert**2)

            logging.info("{} {} tec {} +- {} from {}".format(d, n, tec_mean, tec_uncert, tec_init[d, n]))

            if debug_indicator is not None:
                if debug_indicator(d, n):
                    logging.info("Plotting {} {}".format(d, n))
                    debug_elbo = np.array([ls_obj.loss_func([t, ltu], weights) for t, ltu in zip(debug_tec_range, debug_log_uncert_range)])
                    debug_grid_elbo = griddata(debug_tec_range, debug_log_uncert_range, debug_elbo, *debug_grid_points, interp='linear')
                    plt.imshow(debug_grid_elbo, origin='lower', aspect='auto',
                               extent=(debug_grid_points[0].min(), debug_grid_points[0].max(),debug_grid_points[1].min(), debug_grid_points[1].max()))
                    plt.colorbar()
                    plt.ylabel('log (tec uncertainty[mTECU])')
                    plt.xlabel('tec [mTECU]')
                    xlim = plt.xlim()
                    ylim = plt.ylim()
                    plt.scatter(np.clip(result[0], *xlim), np.clip(result[1], *ylim), c='red')
                    plt.xlim(*xlim)
                    plt.ylim(*ylim)
                    plt.title("Solved {:.1f} +- {:.1f} mTECU".format(result[0], np.exp(result[1])))
                    plt.savefig(os.path.join(output_dir, 'elbo_{:02d}_{:03d}.png'.format(d, n)))
                    plt.close('all')

        ###
        # filter outliers
        # N
        abs_res_real = np.nanmedian(np.abs(Yreal_res[d,:,:]), axis=0)
        # scalar
        mean_abs_res_real = np.nanmedian(abs_res_real, axis=-1)
        std_abs_res_real = np.nanmedian(np.abs(abs_res_real - mean_abs_res_real), axis=-1)
        # N
        flag_real = abs_res_real - mean_abs_res_real > 4. * std_abs_res_real

        # N
        abs_res_imag = np.nanmedian(np.abs(Yimag_res[d, :, :]), axis=0)
        # scalar
        mean_abs_res_imag = np.nanmedian(abs_res_imag, axis=-1)
        std_abs_res_imag = np.nanmedian(np.abs(abs_res_imag - mean_abs_res_imag), axis=-1)
        # N
        flag_imag = abs_res_imag - mean_abs_res_imag > 4. * std_abs_res_imag

        replace_flag = np.logical_or(flag_real, flag_imag)

        keep_flag = np.logical_not(replace_flag)

        keep_idx = np.where(keep_flag)[0]
        replace_idx = np.where(replace_flag)[0]
        flag_array[d, replace_idx] = 1.
        #
        # filtered_mean = median_filter(tec_mean_array[d, :], (5,))
        # filtered_uncert = median_filter(tec_uncert_array[d, :], (5,))
        #
        # tec_mean_array[d, replace_idx] = filtered_mean[replace_idx]
        # tec_uncert_array[d, replace_idx] = filtered_uncert[replace_idx]
    
    return tec_mean_array, tec_uncert_array, flag_array

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
    ampssmoothed = np.exp((median_filter(np.log(amps), size=(1, 1, 1, freqkernel,timekernel), mode='reflect')))
    return ampssmoothed

def distribute_solves(datapack=None, ref_dir_idx=14, num_processes=64, select={'pol':slice(0,1,1)}, plot=False, output_folder='./tec_solve', debug=False):
    output_folder = os.path.abspath(output_folder)
    os.makedirs(output_folder,exist_ok=True)


    datapack = DataPack(datapack, readonly=False)
    logging.info("Creating directionally_referenced/tec000")
    make_soltab(datapack, from_solset='sol000', to_solset='directionally_referenced', from_soltab='phase000', to_soltab=['tec000'])

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
    amp_raw, axes = datapack.amplitude

    datapack.current_solset = 'smoothed000'
    datapack.select(**select)
    smooth_tec, axes = datapack.tec

    amp_model = smoothamps(amp_raw)
    amp_raw = amp_model

    if ref_dir_idx is None:
        ref_dir_idx = np.argmin(np.median(
            np.reshape(np.transpose(np.abs(np.diff(phase_raw, axis=-2)), (1, 0, 2, 3, 4)), (Nd, -1)), axis=1))
        logging.info("Using ref_dir_idx {}".format(ref_dir_idx))
    tec_init = np.clip(smooth_tec - smooth_tec[:, ref_dir_idx:ref_dir_idx + 1, ...], -199., 199.)
    phase_di = phase_raw[:, ref_dir_idx:ref_dir_idx + 1, ...]
    phase_raw = phase_raw - phase_di

    # Npol, Nd, Na, Nf, Nt
    Yimag_full = amp_raw * np.sin(phase_raw)
    Yreal_full = amp_raw * np.cos(phase_raw)

    Yimag_full = Yimag_full.reshape((-1, Nf, Nt))
    Yreal_full = Yreal_full.reshape((-1, Nf, Nt))
    tec_init = tec_init.reshape((-1, Nt))

    if debug:
        debug_indicator = lambda *x: np.random.uniform() < 0.1
    else:
        debug_indicator = lambda *x: False
    D = Yimag_full.shape[0]
    num_processes = min(D, num_processes)
    dsk = {}
    for i in range(0, D, D // num_processes):
        start = i
        stop = min(i + (D // num_processes), D)
        dsk[str(i)] = (sequential_solve, Yreal_full[start:stop, :, :], Yimag_full[start:stop, :, :], freqs, tec_init[start:stop, :], debug_indicator, os.path.join(output_folder, 'debug_proc_{:02d}'.format(i)))
    logging.info("Running dask on {} processes".format(num_processes))
    results = get(dsk, list(dsk.keys()), num_workers=num_processes)
    logging.info("Finished dask")

    tec_mean = np.zeros((D, Nt))
    tec_uncert = np.zeros((D, Nt))

    filter_flags = np.zeros((D, Nt))
    for c, i in enumerate(range(0, D, D // num_processes)):
        start = i
        stop = min(i + (D // num_processes), D)

        tec_mean[start:stop, :] = results[c][0]
        tec_uncert[start:stop, :] = results[c][1]
        filter_flags[start:stop, :] = results[c][2]

    tec_mean = tec_mean.reshape((Npol, Nd, Na, Nt))
    tec_uncert = tec_uncert.reshape((Npol, Nd, Na, Nt))
    filter_flags = filter_flags.reshape((Npol, Nd, Na, Nt))

    filter_flags = np.tile(filter_flags[:, :, :, None, :], (1, 1, 1, Nf, 1))

    logging.info("Storing results in a datapack")
    datapack.current_solset = 'directionally_referenced'
    # Npol, Nd, Na, Nf, Nt
    datapack.select(**select)
    datapack.tec = tec_mean
    datapack.weights_tec = tec_uncert
    logging.info("Stored tec results. Done")


    tec_conv = -8.4479745e6 / freqs
    phase_model = tec_conv[:, None]*tec_mean[..., None,:]

    Yreal_model = amp_raw*np.cos(phase_model)
    Yimag_model = amp_raw*np.sin(phase_model)

    res_real = Yreal_model - Yreal_full.reshape(Yreal_model.shape)
    res_imag = Yimag_model - Yimag_full.reshape(Yimag_model.shape)

    np.savez(os.path.join(output_folder,'residual_data.npz'),res_real = res_real, res_imag = res_imag)
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

                slice_flag = filter_flags[0, j, i, b:b + block_size]

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
                    colors_[np.where(slice_flag)[0]] = np.array([1., 0., 0., 1.])

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
    select = dict(ant=None, time=None, dir=None, freq=None, pol=slice(0, 1, 1))
    distribute_solves('/net/lofar1/data1/albert/imaging/data/lockman/L667218_DDS4_full.h5',
                       ref_dir_idx=0, num_processes=64, select=select, plot=True, debug=False, output_folder='lockman_L667218_tec_vi_run8')
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