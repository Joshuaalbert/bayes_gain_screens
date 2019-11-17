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
from dask.multiprocessing import get
from scipy.optimize import least_squares
import argparse
from timeit import default_timer

"""
This script is still being debugged/tested. 
Smoothes gains.
"""

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

    def __init__(self, Yreal, Yimag, freqs, sigma=None):

        self.tec_conv = -8.4479745e6 / freqs
        self.clock_conv = 2*np.pi * freqs * 1e-9
        # Nf
        self.amp = np.sqrt(np.square(Yreal) + np.square(Yimag))
        self.Yreal = Yreal
        self.Yimag = Yimag

        self.phase = wrap(np.arctan2(Yimag, Yreal))

        self.phase_model = lambda params: self.tec_conv * params[0] + params[1] + self.clock_conv * params[2]
        self.sigma = sigma

    def loss_func(self, params):
        """
        non-VI loss
        """
        tec, const, clock = params[0], params[1], params[2]

        # Nf
        phase = tec * self.tec_conv + const + clock * self.clock_conv
        Yreal_m = self.amp * np.cos(phase)
        Yimag_m = self.amp * np.sin(phase)

        # Nf
        log_prob = log_gaussian_pdf((Yreal_m - self.Yreal),
                                    (Yimag_m - self.Yimag),
                                    self.sigma)

        log_prob = np.sum(log_prob)

        loss = np.negative(log_prob)
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
    plot_period = 120.  # 2 minutes

    D, Nf, N = Yreal.shape

    tec_array = np.zeros((D, N))
    const_array = np.zeros((D, N))
    clock_array = np.zeros((D, N))

    t0 = -np.inf

    for d in range(D):
        dir_idx, ant_idx = unravel_index[0][d], unravel_index[1][d]
        # Nf, Nf N, Nf N
        sigma, Yreal_res, Yimag_res = get_residual_gaussian_statistics(Yimag,
                                                                       Yreal,
                                                                       ant_idx, d, dir_idx)

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
            result = list(brute(ls_obj.loss_func,
                                 (slice(-200., 200., 5.),
                                  slice(-np.pi, np.pi, 0.2),
                                  slice(-0.5, 0.5, 0.1)),
                                 finish=lambda func, x0, **kwargs:minimize(func,x0,method='BFGS').x))

            result[1] = np.arctan2(np.sin(result[1]), np.cos(result[1]))
            tec, const, clock = result
            tec_array[d, n] = tec
            const_array[d, n] = const
            clock_array[d, n] = clock

            with np.printoptions(precision=2):
                logging.info(
                    "dir {:02d} ant {:02d} time {:03d}:\n\ttec {} const {} clock {}".format(dir_idx, ant_idx, n,
                                                                                               tec,
                                                                                               const, clock))

            if default_timer() - t0 > plot_period:
                t0 = default_timer()
                ls_obj = SolveLossMaxLike(Yreal[d, :, n], Yimag[d, :, n], freqs,
                                          sigma=sigma)

                fig, axs = plt.subplots(1, 1, figsize=(6, 6))

                axs[0].set_title(
                    "Solved {:.1f} +- {:.1f} mTECU const {:.2f} +- {:.2f} rad".format(tec_mean, tec_uncert,
                                                                                      const_mean, const_uncert))
                axs[0].plot(freqs, wrap(ls_obj.phase_model(result)), c='red', label='mean phase ML')
                axs[0].legend(frameon=False)
                axs[0].scatter(freqs, ls_obj.phase, label='data')
                axs[0].set_xlabel('freq (Hz)')
                axs[0].set_ylabel('phase (rad)')
                plt.savefig(os.path.join(working_dir, 'phase_res_dir{:02d}_ant{:02d}_time{:03d}.png'.format(dir_idx, ant_idx, n)))
                plt.close('all')

    return tec_array, const_array, clock_array


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


def main(data_dir, working_dir, obs_num, ncpu):
    os.chdir(working_dir)
    logging.info("Performing TEC, constant, and clock smoothing.")
    merged_h5parm = os.path.join(data_dir, 'L{}_{}_merged.h5'.format(obs_num, 'DDS4_full'))
    select = {'pol': slice(0, 1, 1)}
    datapack = DataPack(merged_h5parm, readonly=False)
    logging.info("Creating smoothed000/phase000+amplitude000")
    make_soltab(datapack, from_solset='sol000', to_solset='smoothed000', from_soltab='phase000',
                to_soltab=['phase000', 'amplitude000'])
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
    amp_raw, axes = datapack.amplitude
    logging.info("Smoothing amplitudes in time and frequency.")
    amp_smooth = smoothamps(amp_raw)

    # Npol, Nd, Na, Nf, Nt
    Yimag = amp_smooth * np.sin(phase_raw)
    Yreal = amp_smooth * np.cos(phase_raw)

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
    tec = np.zeros((D, Nt))
    const = np.zeros((D, Nt))
    clock = np.zeros((D, Nt))

    for c, i in enumerate(range(0, D, D // num_processes)):
        start = i
        stop = min(i + (D // num_processes), D)
        tec[start:stop, :] = results[c][0]
        const[start:stop, :] = results[c][1]
        clock[start:stop, :] = results[c][2]

    tec = tec.reshape((Npol, Nd, Na, Nt))
    const = const.reshape((Npol, Nd, Na, Nt))
    clock = clock.reshape((Npol, Nd, Na, Nt))

    tec_conv = -8.4479745e6 / freqs
    clock_conv = 2*np.pi*(freqs*1e-9)
    phase_model = tec_conv[:, None] * tec[..., None, :] + const[..., None, :] + clock_conv[:, None] * clock

    logging.info("Storing results")
    datapack.current_solset = 'smoothed000'
    # Npol, Nd, Na, Nf, Nt
    datapack.select(**select)
    datapack.phase = phase_model
    datapack.amplitude = amp_smooth
    logging.info("Stored phase and amplitude results. Done")

    Yreal_model = amp_smooth * np.cos(phase_model)
    Yimag_model = amp_smooth * np.sin(phase_model)

    res_real = Yreal_model - Yreal.reshape(Yreal_model.shape)
    res_imag = Yimag_model - Yimag.reshape(Yimag_model.shape)

    plot_results(Na, Nd, antenna_labels, working_dir, phase_model, phase_raw, res_imag, res_real)


def plot_results(Na, Nd, antenna_labels, working_dir, phase_model,
                 phase_raw, res_imag, res_real):
    logging.info("Plotting results.")
    for i in range(Na):
        for j in range(Nd):
            slice_phase_data = wrap(phase_raw[0, j, i, :, :])
            slice_phase_model = wrap(phase_model[0, j, i, :, :])
            slice_res_real = res_real[0, j, i, :, :]
            slice_res_imag = res_imag[0, j, i, :, :]
            time_array = np.arange(slice_res_real.shape[-1])
            colors = plt.cm.jet(np.arange(slice_res_real.shape[-1]) / slice_res_real.shape[-1])
            # Nf, Nt
            _slice_res_real = slice_res_real - np.mean(slice_res_real, axis=0)
            _slice_res_imag = slice_res_imag - np.mean(slice_res_imag, axis=0)
            fig, axs = plt.subplots(1, 2, figsize=(15, 15))
            diff_phase = wrap(wrap(slice_phase_data) - wrap(slice_phase_model))
            for nu in range(slice_res_real.shape[-2]):
                f_c = plt.cm.binary((nu + 1) / slice_res_real.shape[-2])
                colors_ = (f_c + colors) / 2. * np.array([1., 1., 1., 1. - (nu + 1) / slice_res_real.shape[-2]])
                axs[0].scatter(time_array, np.abs(slice_res_real[nu, :]), c=colors_, marker='.')
                axs[0].scatter(time_array, -np.abs(slice_res_imag[nu, :]), c=colors_, marker='.')
            axs[0].set_title("Real and Imag residuals")
            axs[0].hlines(0., time_array.min(), time_array.max())
            axs[1].imshow(diff_phase, origin='lower', aspect='auto', cmap='coolwarm', vmin=-0.2,
                             vmax=0.2)
            axs[1].set_title('Phase residuals [-0.2,0.2]')
            axs[1].set_xlabel('Time')
            axs[1].set_ylabel('Freq')
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, 'summary_{}_dir{:02d}.png'.format(antenna_labels[i].decode(), j)))
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Performs variational inference of TEC and a constant per direction, antenna, time and stores in directionally_referenced solset.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_args(parser)
    flags, unparsed = parser.parse_known_args()
    print("Running with:")
    for option, value in vars(flags).items():
        print("    {} -> {}".format(option, value))
    try:
        main(**vars(flags))
        exit(0)
    except:
        exit(1)
