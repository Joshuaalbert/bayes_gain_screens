import os
os.environ['OMP_NUM_THREADS'] = "1"
import numpy as np
import pylab as plt
from scipy.ndimage import median_filter
from bayes_gain_screens import logging
from bayes_gain_screens.datapack import DataPack
from bayes_gain_screens.misc import make_soltab
from scipy.stats import multivariate_normal
from scipy.optimize import minimize
from dask.multiprocessing import get
import argparse
from timeit import default_timer

"""
This script is still being debugged/tested. 
Get's TEC from gains.
"""

def stack_complex(y):
    """
    Stacks complex real and imaginary parts
    :param y: [..., N]
    :return: [...,2N]
    real on top of imag
    """
    return np.stack([y.real, y.imag], axis=-1)

def sequential_solve(Yreal, Yimag, freqs, working_dir):
    """
    Run on blocks of time.

    :param Yreal:
        [D, Nf, Nt]
    :param Yimag:
    :param freqs:
    :return:
        [D, Nf, Nt]
    """

    D, Nf, N = Yreal.shape

    phase_array = np.zeros((D, Nf, N))
    for d in range(D):
        t0 = default_timer()
        Sigma, flag = smooth_gains(Yreal[d, :, :], Yimag[d, :, :], filter_size=1, deg=2)
        Sigma = np.diag(np.diag(Sigma))
        last_params = np.array([0.,0.,0.])

        for t in range(N):
            keep = np.logical_not(flag[:, t])
            Sigma_keep = np.concatenate([keep, keep])
            tec_conv = -8.4479745e6 / freqs[keep]
            flagged_Sigma = Sigma[Sigma_keep, :][:, Sigma_keep]
            clock_conv = 2 * np.pi * (1e-9 * freqs[keep])
            Y_concat = np.concatenate([Yreal[d, keep, t], Yimag[d, keep, t]], axis=0)
            def loss(params):
                tec, const, clock = params
                phase_pred = tec*tec_conv + const + clock*clock_conv
                g_pred = np.concatenate([np.cos(phase_pred), np.sin(phase_pred)],axis=0)
                L = -multivariate_normal(mean=Y_concat, cov=flagged_Sigma).logpdf(g_pred)
                return L
            res = minimize(loss, last_params, method='BFGS')
            last_params = res.x
            tec_conv = -8.4479745e6 / freqs
            clock_conv = 2 * np.pi * (1e-9 * freqs)
            phase_array[d, :, t] = last_params[0]*tec_conv + last_params[1] + last_params[2]*clock_conv
        logging.info("Timing {:.2f} timesteps / second".format(N / (default_timer() - t0)))
    return phase_array


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

def smooth_gains(Yreal, Yimag, filter_size=1, deg=2):
    _Yreal, _Yimag = np.copy(Yreal), np.copy(Yimag)
    Nf, N = Yreal.shape
    _freqs = np.linspace(-1., 1., Nf)
    real = sum([median_filter(p, filter_size) * _freqs[:, None] ** (deg - i) for i, p in
                enumerate(np.polyfit(_freqs, _Yreal, deg=deg))])
    imag = sum([median_filter(p, filter_size) * _freqs[:, None] ** (deg - i) for i, p in
                enumerate(np.polyfit(_freqs, _Yimag, deg=deg))])

    res_real = np.abs(real - Yreal)
    res_imag = np.abs(Yimag - imag)

    res_vec = np.concatenate([res_real, res_imag], axis=0)
    Sigma = res_vec[:, None, :] * res_vec[None, :, :]
    Sigma = np.nanmean(Sigma, axis=-1)

    flagreal = np.abs(real - Yreal) > 4.*np.sqrt(np.diag(Sigma)[:Nf, None])
    flagimag = np.abs(imag - Yimag) > 4.*np.sqrt(np.diag(Sigma)[Nf:, None])

    return Sigma, np.logical_or(flagreal, flagimag)


def main(data_dir, working_dir, obs_num, ncpu):
    os.chdir(working_dir)
    logging.info("Performing TEC and constant variational inference.")
    merged_h5parm = os.path.join(data_dir, 'L{}_DDS4_full_merged.h5'.format(obs_num))
    select = dict(pol=slice(0, 1, 1))
    datapack = DataPack(merged_h5parm, readonly=False)
    logging.info("Creating directionally_referenced/tec000+const000")
    make_soltab(datapack, from_solset='sol000', to_solset='smoothed000', from_soltab='phase000',
                to_soltab=['phase000', 'amplitude000'])
    logging.info("Getting raw phase")
    datapack.current_solset = 'sol000'
    datapack.select(**select)
    axes = datapack.axes_phase
    antenna_labels, antennas = datapack.get_antennas(axes['ant'])
    patch_names, directions = datapack.get_directions(axes['dir'])
    radec = np.stack([directions.ra.rad, directions.dec.rad], axis=1)
    timestamps, times = datapack.get_times(axes['time'])
    freq_labels, freqs = datapack.get_freqs(axes['freq'])
    pol_labels, pols = datapack.get_pols(axes['pol'])
    Npol, Nd, Na, Nf, Nt = len(pols), len(directions), len(antennas), len(freqs), len(times)
    phase_raw, axes = datapack.phase
    amp_raw, axes = datapack.amplitude
    amp_smooth = smoothamps(amp_raw)

    tec_conv = -8.4479745e6 / freqs


    # Npol, Nd, Na, Nf, Nt
    Yimag = amp_smooth * np.sin(phase_raw)
    Yreal = amp_smooth * np.cos(phase_raw)
    Yimag = Yimag.reshape((-1, Nf, Nt))
    Yreal = Yreal.reshape((-1, Nf, Nt))

    logging.info("Building dask")
    D = Yimag.shape[0]
    num_processes = min(D, ncpu)
    dsk = {}
    keys = []
    for c,i in enumerate(range(0, D, D // num_processes)):
        start = i
        stop = min(i + (D // num_processes), D)
        dsk[str(c)] = (sequential_solve, Yreal[start:stop, :, :], Yimag[start:stop, :, :], freqs, working_dir)
        keys.append(str(c))
    logging.info("Running dask on {} processes".format(num_processes))
    results = get(dsk, keys, num_workers=num_processes)
    logging.info("Finished dask.")
    phase_smooth = np.zeros((D, Nf, Nt))
    for c, i in enumerate(range(0, D, D // num_processes)):
        start = i
        stop = min(i + (D // num_processes), D)
        phase_smooth[start:stop, :, :] = results[c][0]

    phase_smooth = phase_smooth.reshape((Npol, Nd, Na, Nf, Nt))

    logging.info("Storing smoothed000/phase000+amplitude000")
    datapack.current_solset = 'smoothed000'
    datapack.select(**select)
    datapack.phase = phase_smooth
    datapack.amplitude = amp_smooth

    plot_results(amp_raw, amp_smooth, freqs, phase_raw, phase_smooth, working_dir)


def plot_results(amp_raw, amp_smooth, freqs, phase_raw, phase_smooth, working_dir):
    logging.info("Plotting some residuals.")
    diff_phase = wrap(wrap(phase_smooth) - wrap(phase_raw))
    diff_amp = np.log(amp_smooth) - np.log(amp_raw)
    worst_ants = np.argsort(np.abs(diff_phase).mean(0).mean(0).mean(-1).mean(-1))[-5:]
    worst_dirs = np.argsort(np.abs(diff_phase).mean(0).mean(-1).mean(-1).mean(-1))[-5:]
    worst_times = np.argsort(np.abs(diff_phase).mean(0).mean(0).mean(0).mean(0))[-5:]
    for d in worst_dirs:
        for a in worst_ants:
            plt.imshow(diff_phase[0, d, a, :, :], origin='lower', aspect='auto', vmin=-0.1, vmax=0.1, cmap='coolwarm')
            plt.xlabel('time')
            plt.ylabel('freq')
            plt.colorbar()
            plt.savefig(os.path.join(working_dir, 'phase_diff_dir{:02d}_ant{:02d}.png'.format(d, a)))
            plt.close('all')
            plt.imshow(diff_amp[0, d, a, :, :], origin='lower', cmap='coolwarm', aspect='auto',
                       vmin=np.percentile(diff_amp[0, d, a, :, :], 5), vmax=np.percentile(diff_amp[0, d, a, :, :], 95))
            plt.xlabel('time')
            plt.ylabel('freq')
            plt.colorbar()
            plt.savefig(os.path.join(working_dir, 'amp_diff_dir{:02d}_ant{:02d}.png'.format(d, a)))
            plt.close('all')
            for t in worst_times:
                plt.scatter(freqs, phase_raw[0, d, a, :, t])
                plt.plot(freqs, phase_smooth[0, d, a, :, t])
                plt.xlabel('freq')
                plt.ylabel('diff phase (rad)')
                plt.savefig(os.path.join(working_dir, 'phase{:02d}_ant{:02d}_time{:03d}.png'.format(d, a, t)))
                plt.close('all')
                plt.scatter(freqs, np.log(amp_raw[0, d, a, :, t]))
                plt.plot(freqs, np.log(amp_smooth[0, d, a, :, t]))
                plt.xlabel('freq')
                plt.ylabel('diff log(amp)')
                plt.savefig(os.path.join(working_dir, 'amp{:02d}_ant{:02d}_time{:03d}.png'.format(d, a, t)))
                plt.close('all')


def wrap(p):
    return np.arctan2(np.sin(p), np.cos(p))



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
        description='Variational inference of DDTEC and a constant term. Updates the smoothed000 solset too.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_args(parser)
    flags, unparsed = parser.parse_known_args()
    print("Running with:")
    for option, value in vars(flags).items():
        print("    {} -> {}".format(option, value))
    main(**vars(flags))
