import os

os.environ['OMP_NUM_THREADS'] = "1"
import matplotlib

matplotlib.use('Agg')
import numpy as np
from scipy.ndimage import median_filter
from bayes_gain_screens import logging
from bayes_gain_screens.datapack import DataPack
from bayes_gain_screens.misc import make_soltab, great_circle_sep, fit_tec_and_noise, rolling_window
import pylab as plt
import argparse

"""
This script is still being debugged/tested. 
Get's TEC from gains.
"""

import tensorflow as tf
from bayes_gain_screens.misc import diagonal_jitter
def smooth(x, Y, deg, cov_prior, sigma, return_resolution=False):
    """
    Smoothes Y assuming the reression coefficients have a zero mean prior and covariance prior `cov_prior`.

    :param x: [N]
    :param Y: [B, N]
    :param deg: int
    :param cov_prior: [B, K, K]
        Ordered from 0..deg
    :param Sigma: [(B), N,N]

    :return:
    """
    K = deg + 1
    X = np.stack([x ** (deg - i) for i in range(K)], axis=0)
    with tf.Session(graph=tf.Graph()) as sess:
        # K,N
        X = tf.constant(X, dtype=tf.float64)
        # K, K
        # cov = tf.constant(cov_prior, dtype=tf.float64)
        # B, K, K
        cov_pl = tf.placeholder(tf.float64, shape=cov_prior.shape)
        # B,N
        y_pl = tf.placeholder(tf.float64, shape=Y.shape)
        B = tf.shape(y_pl)[0]
        N = tf.shape(X)[1]
        # (B), N
        sigma_pl = tf.placeholder(tf.float64, shape=sigma.shape)
        # (B), N
        L_Sigma = sigma_pl
        # B, K, K
        L_O = tf.linalg.cholesky(cov_pl + diagonal_jitter(tf.shape(cov_pl)[-1]))
        # B, K, N
        X_ext = tf.tile(X[None, :, :], [B, 1, 1])
        # B, K, N
        LX = tf.linalg.triangular_solve(L_O, X_ext, adjoint=False, lower=True)
        # B, K,N
        OXT = tf.linalg.triangular_solve(L_O, LX, adjoint=True, lower=True)
        # B, N, N
        XOXT = tf.linalg.matmul(LX, LX, transpose_a=True)
        #B, N, N
        XOXT_ = XOXT/L_Sigma[...,None,:]
        #B, N, N
        _XOXT_ = XOXT_/L_Sigma[..., :, None]
        # B, N, N
        # S = Sigma_pl + XOXT
        # B, N, N
        _S_= tf.eye(N, dtype=tf.float64) + _XOXT_
        # # B, N, N
        # S = tf.tile(S[None, :, :], [tf.shape(y_pl)[0], 1, 1])
        # B, N
        # post_mean = tf.linalg.matmul(XOXT, tf.linalg.lstsq(S, y_pl[:, :, None], fast=False))[:, :, 0]
        _y_pl = y_pl/L_Sigma
        post_mean = tf.linalg.matmul(XOXT_, tf.linalg.lstsq(_S_, _y_pl[:, :, None], fast=False))[:, :, 0]
        post_cov = tf.linalg.matmul(XOXT_, tf.linalg.lstsq(_S_, XOXT/L_Sigma[...,:,None], fast=False))
        post_var = tf.linalg.diag_part(post_cov)
        # B
        # R = tf.linalg.matmul(OXT, tf.linalg.lstsq(S, tf.transpose(X_ext, (0, 2, 1)), fast=False))
        R = tf.linalg.matmul(OXT/L_Sigma[...,None,:],
                             tf.linalg.lstsq(_S_, tf.transpose(X_ext, (0, 2, 1))/L_Sigma[...,:,None], fast=False))

        resolution = tf.linalg.trace(R) / tf.cast(tf.shape(R)[-1], R.dtype)
        if return_resolution:
            return sess.run([post_mean, post_var, resolution], {y_pl: Y, sigma_pl: sigma, cov_pl: cov_prior})
        return sess.run([post_mean, post_var], {y_pl: Y, sigma_pl: sigma, cov_pl: cov_prior})



def get_cov_matrix(freqs, x, deg, tec_scale, with_bias=True, const_scale=1., clock_scale=0.5, N=10000):
    """
    Get expected covariance for gains.

    :param freqs: [Nf]
    :param x: [Nf]
    :param deg: int
    :param tec_scale: [B]
    :param with_bias: bool
    :param bias_scale: float
    :return:
    """
    tec_conv = -8.4479745e6 / freqs
    clock_conv = 2 * np.pi * 1e-9 * freqs
    Nf = freqs.size
    B = len(tec_scale)
    # N,B
    tec = tec_scale * np.random.normal(size=[N, B])
    # Nf, N, B
    phase = tec_conv[:, None, None] * tec
    if with_bias:
        const = np.random.uniform(-const_scale, const_scale, size=[N, B])
        clock = np.random.uniform(-clock_scale, clock_scale, size=[N, B])
        phase += const + clock_conv[:, None, None] * clock

    Yreal = np.cos(phase).reshape((Nf, N * B))
    Yimag = np.sin(phase).reshape((Nf, N * B))
    # K, N, B
    preal = np.polyfit(x, Yreal, deg=deg).reshape((-1, N, B))
    pimag = np.polyfit(x, Yimag, deg=deg).reshape((-1, N, B))
    # # K, N, B
    # preal -= np.mean(preal, axis=1, keepdims=True)
    # pimag -= np.mean(pimag, axis=1, keepdims=True)
    # K, K, B
    cov_real = np.mean(preal[:, None, :, :] * preal[None, :, :, :], axis=-2)
    cov_imag = np.mean(pimag[:, None, :, :] * pimag[None, :, :, :], axis=-2)
    return cov_real, cov_imag


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

def smooth_gains(Yreal, Yimag, freqs, deg=2):
    """

    :param Yreal: [Npol, Nd, Na, Nf, Nt]
    :param Yimag: [Npol, Nd, Na, Nf, Nt]
    :param deg:
    :return:
    """
    Npol, Nd, Na, Nf, Nt = Yreal.shape

    Yreal_out = np.zeros_like(Yreal)
    Yimag_out = np.zeros_like(Yimag)

    stdreal_out = np.zeros_like(Yreal)
    stdimag_out = np.zeros_like(Yimag)

    Nf, N = Yreal.shape
    x = np.linspace(-1., 1., Nf)
    tecs = np.linspace(-300., 300., 601)
    # K, K, B'
    cov_real, cov_imag = get_cov_matrix(freqs, x, deg, tec_scale=tecs, with_bias=True, const_scale=1., clock_scale=0.5)
    for d in range(Nd):
        for a in range(Na):
            # get initial flags for tec fit
            real = sum([p * x[:, None] ** (deg - i) for i, p in
                        enumerate(np.polyfit(x, Yreal[0, d, a, :, :], deg=deg))])
            imag = sum([p * x[:, None] ** (deg - i) for i, p in
                        enumerate(np.polyfit(x, Yimag[0, d, a, :, :], deg=deg))])
            res_real = np.abs(Yreal[0, d, a, :, :] - real)
            flag_real = res_real > np.percentile(res_real, 97.5, axis=1, keepdims=True)
            res_imag = np.abs(Yimag[0, d, a, :, :] - imag)
            flag_imag = res_imag > np.percentile(res_imag, 97.5, axis=1, keepdims=True)
            #Nf, B
            flags = np.logical_or(flag_imag, flag_real)
            # fit tec and get noise and systematics
            # B
            tec, sigma = fit_tec_and_noise(freqs,
                                           Yreal[0, d, a, :, :].T,
                                           Yimag[0, d, a, :, :].T,
                                           step=0.33,
                                           search_n=5,
                                           iter_n=3, flags=flags.T)
            tec_scale = np.std(rolling_window(tec, window=50), axis=-1)
            # replace flags with inf
            sigma_real = np.tile(sigma[None, :Nf], [Nt, 1])
            sigma_real[flag_real] = np.inf
            sigma_imag = np.tile(sigma[None, Nf:], [Nt, 1])
            sigma_imag[flag_imag] = np.inf
            # get prior cov
            # B
            which_cov = np.searchsorted(tecs, tec_scale, side='left')[0]
            # K, K, B
            _cov_real = cov_real[:, :, which_cov]
            _cov_imag = cov_imag[:, :, which_cov]
            # smooth with flags then with reweighted
            _Yreal, _ = smooth(x, Yreal[0, d, a, :, :].T, deg, _cov_real, sigma_real,
                                    return_resolution=False)
            _Yreal, var_real, r_real = smooth(x, Yreal[0, d, a, :, :].T, deg, _cov_real, np.abs(Yreal[0, d, a, :, :].T - _Yreal),
                                              return_resolution=True)
            Yreal_out[0, d, a, :, :] = _Yreal.T
            stdreal_out[0,d, a,:, :] = np.sqrt(var_real.T)
            _Yimag, _ = smooth(x, Yimag[0, d, a, :, :].T, deg, _cov_imag, sigma_imag,
                                    return_resolution=False)
            _Yimag, var_imag, r_imag = smooth(x, Yimag[0, d, a, :, :].T, deg, _cov_imag, np.abs(Yimag[0, d, a, :, :].T - _Yimag),
                                              return_resolution=True)
            Yimag_out[0, d, a, :, :] = _Yimag.T
            stdimag_out[0, d, a, :, :] = np.sqrt(var_imag.T)
            if np.any(r_real < 0.9):
                logging.info("Real smoothing, coefficient determination ill-conditioned: {}".format(np.where(r_real < 0.9)))
            if np.any(r_imag < 0.9):
                logging.info("Imag smoothing, coefficient determination ill-conditioned: {}".format(np.where(r_imag < 0.9)))

    return Yreal_out, Yimag_out, stdreal_out,stdimag_out



def main(data_dir, working_dir, obs_num, smooth_amps, deg):
    working_dir = os.path.abspath(working_dir)
    os.makedirs(working_dir, exist_ok=True)
    data_dir = os.path.abspath(data_dir)
    logging.info("Changed dir to {}".format(working_dir))
    os.chdir(working_dir)
    sol_name = 'DDS4_full'
    datapack = os.path.join(data_dir, 'L{}_{}_merged.h5'.format(obs_num, sol_name))
    if not os.path.isfile(datapack):
        raise IOError("datapack doesn't exists {}".format(datapack))

    logging.info("Using working directory: {}".format(working_dir))
    select = dict(pol=slice(0, 1, 1))

    datapack = DataPack(datapack, readonly=False)
    logging.info("Creating smoothed/phase000+amplitude000")
    make_soltab(datapack, from_solset='sol000', to_solset='smoothed000', from_soltab='phase000',
                to_soltab=['phase000', 'amplitude000'], remake_solset=True)
    logging.info("Getting phase and amplitude data")
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
    if smooth_amps:
        logging.info("Smoothing amplitudes in time and frequency.")
        amp_smooth = smoothamps(amp_raw)
    else:
        amp_smooth = amp_raw
    logging.info("Constructing gains.")
    Yimag = amp_smooth * np.sin(phase_raw)
    Yreal = amp_smooth * np.cos(phase_raw)
    Yreal_smooth, Yimag_smooth, stdreal, stdimag = smooth_gains(Yreal, Yimag, freqs, deg=deg)
    phase_smooth = np.arctan2(Yimag_smooth, Yreal_smooth)
    if not smooth_amps:
        amp_smooth = np.sqrt(Yreal_smooth**2 + Yimag_smooth**2)

    # amp_smooth =  np.sqrt(Yimag_full**2 + Yreal_full**2)
    logging.info("Storing results in a datapack")
    datapack.current_solset = 'smoothed000'
    # Npol, Nd, Na, Nf, Nt
    datapack.select(**select)
    datapack.phase = phase_smooth
    datapack.amplitude = amp_smooth
    logging.info("Plotting some residuals.")
    diff_phase = wrap(phase_smooth - wrap(phase_raw))
    diff_amp = np.log(amp_smooth) - np.log(amp_raw)
    worst_ants = np.argsort(np.square(diff_phase).mean(0).mean(0).mean(-1).mean(-1))[-10:]
    worst_dirs = np.argsort(np.square(diff_phase).mean(0).mean(-1).mean(-1).mean(-1))[-10:]
    worst_times = np.argsort(np.square(diff_phase).mean(0).mean(0).mean(0).mean(0))[-10:]

    for d in worst_dirs:
        for a in worst_ants:
            plt.imshow(diff_phase[0, d, a, :, :], origin='lower', aspect='auto', vmin=-0.1, vmax=0.1, cmap='coolwarm')
            plt.colorbar()
            plt.xlabel('time')
            plt.ylabel('freq')
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


def add_args(parser):
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument('--obs_num', help='Obs number L*',
                        default=None, type=int, required=True)
    parser.add_argument('--data_dir', help='Where are the ms files are stored.',
                        default=None, type=str, required=True)
    parser.add_argument('--working_dir', help='Where to perform the imaging.',
                        default=None, type=str, required=True)
    parser.add_argument('--smooth_amps', help='Whether to smooth amps before smoothing gains.',
                        default=True, type="bool", required=False)
    parser.add_argument('--deg', help='Which degree.',
                        default=2, type=int, required=False)


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
