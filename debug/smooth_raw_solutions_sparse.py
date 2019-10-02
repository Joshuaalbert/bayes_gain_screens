import numpy as np
import pylab as plt
import os
from scipy.optimize import brute, fmin
from scipy.ndimage import median_filter
from scipy.sparse import block_diag, coo_matrix
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

    def __init__(self, Yreal, Yimag, freqs, S=20, L=None):

        self.D, self.Nf = Yreal.shape
        self.x, self.w = np.polynomial.hermite.hermgauss(S)
        self.w /= np.pi

        self.tec_conv = -8.4479745e6 / freqs
        self.clock_conv = 2 * np.pi * freqs * 1e-9
        # Nf
        self.amp = np.sqrt(np.square(Yreal) + np.square(Yimag))
        self.Yreal = Yreal
        self.Yimag = Yimag

        self.loss_func = self._least_sqaures_loss_func_corr
        #D, Nf, 2, 2
        self.L = L
        # print(self.amp / np.median(self.amp))

    def phase_model(self, params):
        params = params.reshape((self.D, 3))
        return self.tec_conv * params[:,0,None] + self.clock_conv * params[:,1,None] + params[:,2,None]

    def calculate_residuals(self, params):
        """

        :param params: [D*3]
        :return: [D, Nf]
        """
        phase = self.phase_model(params)
        Yreal_m = self.amp * np.cos(phase)
        Yimag_m = self.amp * np.sin(phase)
        res_real = np.abs(self.Yreal - Yreal_m)
        res_imag = np.abs(self.Yimag - Yimag_m)
        return res_real, res_imag

    def _least_sqaures_loss_func_uncorr(self, params, weights=None):
        """
        least squares loss

        :param params: [D*3]
        :param weights: [D, Nf]
        :return: [D]
        """
        # D, Nf
        phase = self.phase_model(params)
        Yreal_m = self.amp * np.cos(phase)
        Yimag_m = self.amp * np.sin(phase)
        # D, Nf
        res_real = Yreal_m - self.Yreal
        res_imag = Yimag_m - self.Yimag
        # [2, D, Nf]
        if weights is not None:
            return np.mean((np.square(res_real, out=res_real) + np.square(res_imag, out=res_imag))*weights, axis=-1)
        else:
            return np.mean((np.square(res_real, out=res_real) + np.square(res_imag, out=res_imag)), axis=-1)

    def _least_sqaures_loss_func_corr(self, params, weights=None):
        """
        least squares loss

        :param params: [D*3]
        :param weights: [D, Nf]
        :return: [D]
        """
        # D, Nf
        phase = self.phase_model(params)
        Yreal_m = self.amp * np.cos(phase)
        Yimag_m = self.amp * np.sin(phase)
        # D, Nf
        res_real = Yreal_m - self.Yreal
        res_imag = Yimag_m - self.Yimag
        # [2, D, Nf]
        A = special_solve_triangular(self.L, res_real, res_imag)
        if weights is not None:
            return np.mean(np.sum(np.square(A, out=A), axis=0)*weights, axis=-1)
        else:
            return np.mean(np.sum(np.square(A, out=A), axis=0), axis=-1)

def batched_cholesky(rr,ii, ri, out_L=None):
    """
    Created batched cholesky
    :param rr: [...]
    :param ii: [...]
    :param ri: [...]
    :return: [2,2 ...]
    """
    if out_L is None:
        out_L = np.zeros((2,2) + rr.shape, dtype=rr.dtype)
    l00 = np.sqrt(rr, out=out_L[0,0,...])
    l10 = np.divide(ri,l00,out=out_L[1,0,...])
    l11 = np.sqrt(ii - l10**2, out=out_L[1,1,...])
    return out_L

def special_solve_triangular(L, b0, b1, out_x=None):
    """
    solves 2x2 triangular solve. batched
    :param L: [2,2...]
    :param b0: [...]
    :param b1: [...]
    :return: [2, ...] L\b
    """
    if out_x is None:
        out_x = np.zeros(L.shape[1:], dtype=L.dtype)
    x1 = np.divide(b0, L[0,0,...], out=out_x[0, ...])
    x2 = np.divide(b1 - L[1,0, ...]*x1, L[1,1,...], out=out_x[1,...])
    return out_x

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

    least_squares_init = np.zeros((D,3)).reshape((-1,))
    #2,2,D,Nf
    L = 0.1*np.tile(np.eye(2)[:, :, None,None], (1, 1, D, Nf))

    bounds = [np.stack([-200.*np.ones(D), -np.inf*np.ones(D), -np.pi*np.ones(D)], axis=1).reshape((-1,)),
              np.stack([200.*np.ones(D), np.inf*np.ones(D), np.pi*np.ones(D)], axis=1).reshape((-1,))]

    jac_sparsity = np.zeros((D, D, 3))
    for d in range(D):
        jac_sparsity[d, d,:] = 1.
    # D, D*3
    jac_sparsity = jac_sparsity.reshape((D, D*3))
    jac_sparsity = coo_matrix(jac_sparsity)
    stat_block_size = 10
    for n in range(N):

        if n >= stat_block_size:
            #D, Nf, stat_block_size
            _res_real = Yreal_res[:, :, n - stat_block_size:n] - np.mean(Yreal_res[:, :, n - stat_block_size:n], axis=-1, keepdims=True)
            _res_imag = Yimag_res[:, :, n - stat_block_size:n] - np.mean(Yimag_res[:, :, n - stat_block_size:n], axis=-1 ,keepdims=True)
            #D, Nf
            rr = np.mean(_res_real**2, axis=-1)
            ii = np.mean(_res_imag**2, axis=-1)
            ri = np.mean(_res_real*_res_imag, axis=-1)
            L = batched_cholesky(rr, ii, ri, L)

        # least square two pass
        ls_obj = SolveLoss(Yreal[:, :, n], Yimag[:, :, n], freqs, L=L)
        res0 = least_squares(ls_obj.loss_func, least_squares_init, 
                             bounds=bounds, jac_sparsity=jac_sparsity,
                             x_scale='jac', ftol=1e-09, xtol=1e-09)
        
        # compute covariance estimate
        # D, Nf
        res_real0, res_imag0 = ls_obj.calculate_residuals(res0.x)

        ###
        # flag
        if n >= stat_block_size:
            abs_res_real0, abs_res_imag0 = np.abs(res_real0), np.abs(res_imag0)

            #D, Nf, stat_block_size
            abs_res_real = np.abs(Yreal_res[:, :, n - stat_block_size:n])
            abs_res_imag = np.abs(Yimag_res[:, :, n - stat_block_size:n])
            
            ###
            # flag bad frequencies
            #D, Nf
            mean_abs_res_real = np.mean(abs_res_real, axis=-1)
            std_abs_res_real = np.std(abs_res_real, axis=-1)
            mean_abs_res_imag = np.mean(abs_res_imag, axis=-1)
            std_abs_res_imag = np.std(abs_res_imag, axis=-1)
            #D, Nf
            weights = np.logical_and(
                abs_res_real0 - mean_abs_res_real < 2.5 * std_abs_res_real,
                abs_res_imag0 - mean_abs_res_imag < 2.5 * std_abs_res_imag)

            over_flag = np.sum(weights, axis=-1) < freqs.size // 2
            if np.any(over_flag):
                #D, Nf
                over_flag = np.tile(over_flag[:, None], (1, Nf))
                #D, Nf
                total_res = abs_res_imag0 + abs_res_real0
                weights = np.where(over_flag, total_res < np.median(total_res), weights)

            over_flag = np.sum(weights, axis=-1) == 0
            if np.any(over_flag):
                # D, Nf
                over_flag = np.tile(over_flag[:, None], (1, Nf))
                # D, Nf
                weights = np.where(over_flag, 1., weights)
        else:
            weights = np.ones((D, Nf))

        ls_obj = SolveLoss(Yreal[:, :, n], Yimag[:, :, n], freqs, L=L)
        res1 = least_squares(ls_obj.loss_func, res0.x, args=(weights,),
                             bounds=bounds, jac_sparsity=jac_sparsity,
                             x_scale='jac', ftol=1e-09, xtol=1e-09)
        #D, Nf
        res_real1, res_imag1 = ls_obj.calculate_residuals(res1.x)

        new_params = np.reshape(res1.x, (D, 3))
        least_squares_init = res1.x

        Yreal_res[:, :, n] = res_real1
        Yimag_res[:, :, n] = res_imag1

        tec_mean_array[:, n] = new_params[:,0]
        clock_mean_array[:, n] = new_params[:,1]
        const_mean_array[:, n] = new_params[:,2]
        
        logging.info("Done step: {}".format(n))

        # logging.info("{} {} tec {} clock {} const {}".format(
        #     d, n, tec_mean, clock_mean, const_mean ))

    ###
    # filter outliers
    #D, N
    abs_Yreal_res = np.mean(np.abs(Yreal_res), axis=1)
    abs_Yimag_res = np.mean(np.abs(Yimag_res), axis=1)
    
    #D
    mean_abs_res_real = np.mean(abs_Yreal_res, axis=-1)
    std_abs_res_real = np.std(abs_Yreal_res, axis=-1)
    mean_abs_res_imag = np.mean(abs_Yimag_res, axis=-1)
    std_abs_res_imag = np.std(abs_Yimag_res, axis=-1)

    # D, N
    flag_real = abs_Yreal_res - mean_abs_res_real > 2.5 * std_abs_res_real
    flag_imag = abs_Yimag_res - mean_abs_res_imag > 2.5 * std_abs_res_imag

    replace_flag = np.logical_or(flag_real, flag_imag)


    keep_flag = np.logical_not(replace_flag)

    keep_idx = np.where(keep_flag)
    replace_idx = np.where(replace_flag)
    tec_mean_array[replace_idx] = median_filter(tec_mean_array, (1, 9))[replace_idx]
    clock_mean_array[replace_idx] = median_filter(clock_mean_array, (1, 9))[replace_idx]
    const_mean_array[replace_idx] = median_filter(const_mean_array, (1, 9))[replace_idx]

    tec_mean_array[replace_idx] = median_filter(tec_mean_array, (1, 9))[replace_idx]
    clock_mean_array[replace_idx] = median_filter(clock_mean_array, (1, 9))[replace_idx]
    const_mean_array[replace_idx] = median_filter(const_mean_array, (1, 9))[replace_idx]

    flag_array[replace_idx] = 1.

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
    select = dict(ant=slice(57, None, 1), time=slice(0, 10, 1), dir=slice(0,1,1), freq=None, pol=slice(0, 1, 1))
    distribute_solves('/net/lofar1/data1/albert/imaging/data/lockman/L667218_DDS4_full.h5',
                      ref_dir_idx=0, num_processes=1, select=select,
                      output_folder='lockman_L667218_better_amps_sparse', plot=True)
    # distribute_solves('/net/lofar1/data1/albert/imaging/data/P126+65_compact_raw/P126+65_full_compact_raw_v11.h5',
    #                   ref_dir_idx=14, num_processes=56, select=select,
    #                   output_folder='test_smooth')
    # distribute_solves('/home/albert/lofar1_1/imaging/data/LBA/zwcl0634_ddf_kms_15dir_2min.npz',#'/net/bovenrijn/data1/wwilliams/lba_test/zwcl0634/ddf_kMS_15dir/DDS5_2min_full_merged.h5',
    #                   ref_dir_idx=None, num_processes=64, numpy_data=True, elbo_save='/home/albert/lofar1_1/imaging/data/LBA/zwcl0634_ddf_kms_15dir_2min_neg_elbo.npz')
    # distribute_solves('/home/albert/lofar1_1/imaging/data/LBA/zwcl0634_ddf_kms_15dir_1min.npz',#'/net/bovenrijn/data1/wwilliams/lba_test/zwcl0634/ddf_kMS_15dir/DDS5_1min_full_merged.h5',
    #                   ref_dir_idx=0, num_processes=64, select=select,
    #                   output_folder='lba_1min', numpy_data=True)