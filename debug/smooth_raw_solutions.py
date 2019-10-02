import numpy as np
import pylab as plt
import os
from scipy.optimize import brute, fmin
from scipy.ndimage import median_filter
from scipy.linalg import solve_triangular
from bayes_gain_screens import logging
from bayes_gain_screens.datapack import DataPack
from bayes_gain_screens.misc import make_soltab
from dask.multiprocessing import get
from collections import deque
from scipy.optimize import least_squares
from scipy.interpolate import interp1d

"""
This script is still being debugged/tested. 
Get's TEC from gains.
"""


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
                 S=20, L=None):
        self.x, self.w = np.polynomial.hermite.hermgauss(S)
        self.w /= np.pi

        self.tec_conv = -8.4479745e6 / freqs
        self.clock_conv = 2 * np.pi * freqs * 1e-9
        # Nf
        self.amp = np.sqrt(np.square(Yreal) + np.square(Yimag))
        self.Yreal = Yreal
        self.Yimag = Yimag


        self.tec_mean_prior = tec_mean_prior
        self.tec_uncert_prior = tec_uncert_prior


        self.phase_model = lambda params: self.tec_conv * params[0] + self.clock_conv * params[1] + params[2]
        self.loss_func = self._least_sqaures_loss_func
        self.L = L
        # print(self.amp / np.median(self.amp))

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
        log_norm = -0.5 * np.log(2 * np.pi) - np.log(uncert_prior) - 0.5 * (mean - mean_prior) ** 2 / uncert_prior ** 2
        return -log_norm



    def _least_sqaures_loss_func(self, params, weights=None):
        """
        VI loss
        :param params: tf.Tensor
            shapfrom scipy.optimize import brute, fmine [D]
        :return: tf.Tensor
            scalar The loss
        """
        tec_mean, clock_mean, const_mean = params
        if weights is None:
            weights = np.ones(self.tec_conv.size)
        weights = weights.astype(tec_mean.dtype)

        #
        tec = tec_mean
        clock = clock_mean
        const = const_mean

        # Nf
        phase = tec * self.tec_conv + clock * self.clock_conv + const
        Yreal_m = self.amp * np.cos(phase)
        Yimag_m = self.amp * np.sin(phase)
        # 2, Nf
        res = np.stack([Yreal_m - self.Yreal, Yimag_m - self.Yimag], axis=0)
        # 2, Nf
        A = solve_triangular(self.L, res)
        return np.mean(np.sum(np.square(A), axis=0) * weights)



def post_mean(X, Y, sigma_y, amp, l, X2):
    def _kern(x1, x2):
        dx = -0.5 * np.square((x1[:, None] - x2[None, :]) / l)
        return np.exp(2 * np.log(amp) + dx)

    K = _kern(X, X)
    Ks = _kern(X, X2)
    L = np.linalg.cholesky(K + sigma_y ** 2 * np.eye(X.shape[0]))
    A = solve_triangular(L, Ks, lower=True)
    return A.T.dot(solve_triangular(L, Y[:, None], lower=True))[:, 0]


def sequential_solve(Yreal, Yimag, freqs):
    """
    Run on blocks of time.

    :param Yreal:
        [D, Nf, Nt]
    :param Yimag:
    :param freqs:
    :return:
        [D, Nt], [D, Nt]
    """

    D, Nf, N = Yreal.shape

    Yreal_res = np.zeros_like(Yreal)
    Yimag_res = np.zeros_like(Yimag)

    tec_mean_array = np.zeros((D, N))
    clock_mean_array = np.zeros((D, N))
    const_mean_array = np.zeros((D, N))

    flag_array = np.zeros((D, N))
    for d in range(D):

        least_squares_init = [0., 0., 0.]
        L = np.eye(2) * 0.1

        for n in range(N):

            # least square two pass
            ls_obj = SolveLoss(Yreal[d, :, n], Yimag[d, :, n], freqs,
                                L=L)
            res0 = least_squares(ls_obj.loss_func, least_squares_init,
                                 bounds=[[-200., -np.inf, -np.pi], [200., np.inf, np.pi]])
            # compute covariance estimate
            # Nf
            res_real0, res_imag0 = ls_obj.calculate_residuals(res0.x)

            if n > 10:
                ###
                # flag bad frequencies
                mean_abs_res_real = np.mean(np.abs(Yreal_res[d, :, :n + 1]))
                std_abs_res_real = np.std(np.abs(Yreal_res[d, :, :n + 1]))
                mean_abs_res_imag = np.mean(np.abs(Yimag_res[d, :, :n + 1]))
                std_abs_res_imag = np.std(np.abs(Yimag_res[d, :, :n + 1]))

                weights = np.logical_not(
                    np.logical_or(np.abs(res_real0) - mean_abs_res_real > 2.5 * std_abs_res_real,
                                  np.abs(res_imag0) - mean_abs_res_imag > 2.5 * std_abs_res_imag))
                if weights.sum() < freqs.size // 2:
                    weights = np.abs(res_real0) + np.abs(res_imag0) < np.median(np.abs(res_real0) + np.abs(res_imag0))
            else:
                weights = np.ones(Nf)

            keep = np.where(weights)[0]
            res_real0 = res_real0[keep]
            res_imag0 = res_imag0[keep]
            res_real0, res_imag0 = res_real0 - np.mean(res_real0), res_imag0 - np.mean(res_imag0)
            rr = np.mean(res_real0 ** 2)
            ii = np.mean(res_imag0 ** 2)
            ri = np.mean(res_real0 * res_imag0)
            cov = np.array([[rr, ri], [ri, ii]])
            try:
                L = np.linalg.cholesky(cov)
            except:
                pass
            # round two
            ls_obj = SolveLoss(Yreal[d, :, n], Yimag[d, :, n], freqs,
                               L=L)

            res1 = least_squares(ls_obj.loss_func, least_squares_init,
                                 bounds=[[-200., -np.inf, -np.pi], [200., np.inf, np.pi]], args=(weights,))
            new_params = res1.x
            least_squares_init = new_params

            res_real1, res_imag1 = ls_obj.calculate_residuals(res1.x)

            Yreal_res[d, :, n] = res_real1
            Yimag_res[d, :, n] = res_imag1

            tec_mean, clock_mean, const_mean = new_params
            tec_mean_array[d, n] = tec_mean
            clock_mean_array[d, n] = clock_mean
            const_mean_array[d, n] = const_mean

            logging.info("{} {} tec {} clock {} const {}".format(
                d, n, tec_mean, clock_mean, const_mean ))

        ###
        # filter outliers
        # N
        abs_res_real = np.mean(np.abs(Yreal_res[d, :, :]), axis=0)
        # scalar
        mean_abs_res_real = np.mean(abs_res_real, axis=-1)
        std_abs_res_real = np.std(abs_res_real, axis=-1)
        # N
        flag_real = abs_res_real - mean_abs_res_real > 3. * std_abs_res_real

        # N
        abs_res_imag = np.mean(np.abs(Yimag_res[d, :, :]), axis=0)
        # scalar
        mean_abs_res_imag = np.mean(abs_res_imag, axis=-1)
        std_abs_res_imag = np.std(abs_res_imag, axis=-1)
        # N
        flag_imag = abs_res_imag - mean_abs_res_imag > 3. * std_abs_res_imag

        replace_flag = np.logical_or(flag_real, flag_imag)

        keep_flag = np.logical_not(replace_flag)

        keep_idx = np.where(keep_flag)[0]
        replace_idx = np.where(replace_flag)[0]
        flag_array[d, replace_idx] = 1.

        # if len(replace_idx) > 0:
        #     ###
        #     # Simple GP replacement
        #     replace_tec = post_mean(keep_idx, tec_mean_array[d, keep_idx],
        #               np.sqrt(np.mean(np.diff(tec_mean_array[d, keep_idx])**2)), np.std(tec_mean_array[d, keep_idx]),
        #               10., replace_idx)
        #     tec_mean_array[d, replace_idx] = replace_tec
        #     replace_clock = post_mean(keep_idx, clock_mean_array[d, keep_idx],
        #                             np.sqrt(np.mean(np.diff(clock_mean_array[d, keep_idx]) ** 2)),
        #                             np.std(clock_mean_array[d, keep_idx]),
        #                             10., replace_idx)
        #     clock_mean_array[d, replace_idx] = replace_clock
        #     replace_const = post_mean(keep_idx, const_mean_array[d, keep_idx],
        #                             np.sqrt(np.mean(np.diff(const_mean_array[d, keep_idx]) ** 2)),
        #                             np.std(const_mean_array[d, keep_idx]),
        #                             10., replace_idx)
        #     const_mean_array[d, replace_idx] = replace_const
        #
        #     # tec_interp = interp1d(keep_idx, tec_mean_array[d, keep_idx], kind='linear', axis=-1, fill_value="extrapolate",
        #     #                       assume_sorted=True)
        #     # clock_interp = interp1d(keep_idx, clock_mean_array[d, keep_idx], kind='linear', axis=-1, fill_value="extrapolate",
        #     #                         assume_sorted=True)
        #     # const_interp = interp1d(keep_idx, const_mean_array[d, keep_idx], kind='linear', axis=-1, fill_value="extrapolate",
        #     #                         assume_sorted=True)
        #     #
        #     # replace_tec = tec_interp(replace_idx)
        #     # replace_clock = clock_interp(replace_idx)
        #     # replace_const = const_interp(replace_idx)
        #     # print(tec_mean_array[d, replace_idx])
        #     # print(clock_mean_array[d, replace_idx])
        #     # print(const_mean_array[d, replace_idx])
        #     # tec_mean_array[d,replace_idx] = replace_tec
        #     # clock_mean_array[d,replace_idx] = replace_clock
        #     # const_mean_array[d,replace_idx] = replace_const
        #     # print(tec_mean_array[d, replace_idx])
        #     # print(clock_mean_array[d, replace_idx])
        #     # print(const_mean_array[d, replace_idx])

    return tec_mean_array, clock_mean_array, const_mean_array, flag_array


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

def distribute_solves(datapack=None, ref_dir_idx=14, num_processes=64, numpy_data=False,
                       select={'pol': slice(0, 1, 1)}, plot=False, output_folder='./smooth_output'):
    output_folder = os.path.abspath(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    if numpy_data:
        np_datapack = np.load(datapack)
        phase_raw, amp_raw, freqs = np_datapack['phase'], np_datapack['amplitude'], np_datapack['freqs']
        Npol, Nd, Na, Nf, Nt = phase_raw.shape
    else:
        datapack = DataPack(datapack, readonly=False)
        make_soltab(datapack, from_solset='sol000', to_solset='smoothed000', from_soltab='phase000',
                    to_soltab=['tec000', 'phase000', 'amplitude000'])

        logging.info("Creating smoothed000/[tec000,phase000,amplitude000]")
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

    amp_model = smoothamps(amp_raw)
    amp_raw = amp_model

    if ref_dir_idx is None:
        ref_dir_idx = np.argmin(
            np.median(np.reshape(np.transpose(np.abs(np.diff(phase_raw, axis=-2)), (1, 0, 2, 3, 4)), (Nd, -1)), axis=1))
        logging.info("Using ref_dir_idx {}".format(ref_dir_idx))
    phase_di = 0. * phase_raw[:, ref_dir_idx:ref_dir_idx + 1, ...]
    phase_raw = phase_raw - phase_di

    # Npol, Nd, Na, Nf, Nt
    Yimag_full = amp_raw * np.sin(phase_raw)
    Yreal_full = amp_raw * np.cos(phase_raw)

    Yimag_full = Yimag_full.reshape((-1, Nf, Nt))
    Yreal_full = Yreal_full.reshape((-1, Nf, Nt))

    D = Yimag_full.shape[0]
    dsk = {}
    for i in range(0, D, D // num_processes):
        start = i
        stop = min(i + (D // num_processes), D)
        dsk[str(i)] = (sequential_solve, Yreal_full[start:stop, :, :], Yimag_full[start:stop, :, :], freqs)
    logging.info("Running dask on {} processes".format(num_processes))
    results = get(dsk, list(dsk.keys()), num_workers=num_processes)
    logging.info("Finished dask")

    tec_mean = np.zeros((D, Nt))
    tec_uncert = np.zeros((D, Nt))
    clock_mean = np.zeros((D, Nt))
    const_mean = np.zeros((D, Nt))
    filter_flags = np.zeros((D, Nt))
    num_processes = min(D, num_processes)
    for c, i in enumerate(range(0, D, D // num_processes)):
        start = i
        stop = min(i + (D // num_processes), D)


        tec_mean[start:stop, :] = results[c][0]
        clock_mean[start:stop, :] = results[c][1]
        const_mean[start:stop, :] = results[c][2]
        filter_flags[start:stop, :] = results[c][3]

    tec_mean = tec_mean.reshape((Npol, Nd, Na, Nt))
    tec_uncert = tec_uncert.reshape((Npol, Nd, Na, Nt))  # not filled out
    clock_mean = clock_mean.reshape((Npol, Nd, Na, Nt))
    const_mean = const_mean.reshape((Npol, Nd, Na, Nt))
    filter_flags = filter_flags.reshape((Npol, Nd, Na, Nt))

    tec_conv = -8.4479745e6 / freqs
    clock_conv = 2 * np.pi * freqs * 1e-9
    phase_model = tec_conv[:, None] * tec_mean[..., None, :] + clock_conv[:, None] * clock_mean[..., None,:] \
                  + const_mean[..., None, :]

    filter_flags = np.tile(filter_flags[:, :, :, None, :], (1, 1, 1, Nf, 1))

    # phase_model = np.where(filter_flags == 0., phase_model, phase_raw)

    if numpy_data:
        np.savez(datapack, phase=phase_raw, amplitude=amp_raw, freqs=freqs,
                 tec_mean=tec_mean, clock_mean=clock_mean,
                 const_mean=const_mean)
    else:
        logging.info("Storing results in a datapack")
        datapack.current_solset = 'smoothed000'
        # Npol, Nd, Na, Nf, Nt
        datapack.select(**select)
        datapack.phase = phase_model
        datapack.amplitude = amp_model
        datapack.tec = tec_mean
        datapack.weights_tec = tec_uncert
        logging.info("Stored tec results. Done")


    Yreal_model = amp_raw * np.cos(phase_model)
    Yimag_model = amp_raw * np.sin(phase_model)

    res_real = Yreal_model - Yreal_full.reshape(Yreal_model.shape)
    res_imag = Yimag_model - Yimag_full.reshape(Yimag_model.shape)



    np.savez(os.path.join(output_folder, 'residual_data.npz'), res_real=res_real, res_imag=res_imag)
    print('./residual_data.npz')

    os.makedirs(os.path.join(output_folder, 'summary'), exist_ok=True)
    if plot:
        block_size = 320
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
                    c_time = time_array / np.max(time_array)
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
                    slice_clock = clock_mean[0, j, i, b:b + block_size]
                    slice_const = const_mean[0, j, i, b:b + block_size]
                    fig, axs = plt.subplots(4, 3, figsize=(24, 20))

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
                        axs[3][1].scatter(slice_real[nu, :], slice_imag[nu, :], c=colors_, marker='.')
                    axs[0][0].set_title("Real and Imag residuals")
                    axs[0][0].hlines(0., time_array.min(), time_array.max())
                    axs[3][1].set_title("Real and Imag model")
                    axs[3][1].set_xlim(-1.4, 1.4)
                    axs[3][1].set_ylim(-1.4, 1.4)
                    axs[0][1].scatter(time_array, np.sqrt(rr), c=colors)
                    axs[0][1].set_title(r'$\sigma_{\rm real}$')
                    axs[1][0].scatter(time_array, np.sqrt(ii), c=colors)
                    axs[1][0].set_title(r'$\sigma_{\rm imag}$')
                    axs[1][1].scatter(time_array, rho, c=colors)
                    axs[1][1].set_title(r'$\rho$')
                    axs[2][0].scatter(time_array, slice_tec, c=colors)
                    axs[2][1].scatter(time_array, slice_clock, c=colors)
                    axs[3][0].scatter(time_array, slice_const, c=colors)
                    axs[2][0].set_title("TEC")
                    axs[2][1].set_title("Clock")
                    axs[3][0].set_title("Const")
                    plt.tight_layout()

                    plt.savefig(os.path.join(output_folder, 'summary',
                                             'fig_{}_{}_{}.png'.format(antenna_labels[i].decode(), j, b)))
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
                      ref_dir_idx=0, num_processes=72, select=select,
                      output_folder='lockman_L667218_better_amps')
    # distribute_solves('/net/lofar1/data1/albert/imaging/data/P126+65_compact_raw/P126+65_full_compact_raw_v11.h5',
    #                   ref_dir_idx=14, num_processes=56, select=select,
    #                   output_folder='test_smooth')
    # distribute_solves('/home/albert/lofar1_1/imaging/data/LBA/zwcl0634_ddf_kms_15dir_2min.npz',#'/net/bovenrijn/data1/wwilliams/lba_test/zwcl0634/ddf_kMS_15dir/DDS5_2min_full_merged.h5',
    #                   ref_dir_idx=None, num_processes=64, numpy_data=True, elbo_save='/home/albert/lofar1_1/imaging/data/LBA/zwcl0634_ddf_kms_15dir_2min_neg_elbo.npz')
    # distribute_solves('/home/albert/lofar1_1/imaging/data/LBA/zwcl0634_ddf_kms_15dir_1min.npz',#'/net/bovenrijn/data1/wwilliams/lba_test/zwcl0634/ddf_kMS_15dir/DDS5_1min_full_merged.h5',
    #                   ref_dir_idx=0, num_processes=64, select=select,
    #                   output_folder='lba_1min', numpy_data=True)