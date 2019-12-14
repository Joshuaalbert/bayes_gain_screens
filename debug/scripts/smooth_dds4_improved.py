import os
os.environ['OMP_NUM_THREADS'] = "1"
import matplotlib
matplotlib.use('Agg')
import numpy as np
from scipy.ndimage import median_filter
from bayes_gain_screens import logging
from bayes_gain_screens.datapack import DataPack
from bayes_gain_screens.misc import make_soltab, great_circle_sep
import pylab as plt
import argparse

"""
This script is still being debugged/tested. 
Get's TEC from gains.
"""

import tensorflow as tf
from bayes_gain_screens.misc import diagonal_jitter


def smooth(x, Y, deg, cov_prior, Sigma, return_resolution=False):
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
        # N,N
        Sigma_pl = tf.placeholder(tf.float64, shape=Sigma.shape)
        # B, K, K
        L_O = tf.linalg.cholesky(cov_pl + diagonal_jitter(tf.shape(cov_pl)[-1]))
        #B, K, N
        X_ext = tf.tile(X[None, :, :], [B, 1, 1])
        # B, K, N
        LX = tf.linalg.triangular_solve(L_O, X_ext, adjoint=False, lower=True)
        # B, K,N
        OXT = tf.linalg.triangular_solve(L_O, LX, adjoint=True, lower=True)
        # B, N, N
        XOXT = tf.linalg.matmul(LX, LX, transpose_a=True)
        # B, N, N
        S = Sigma_pl + XOXT
        # # B, N, N
        # S = tf.tile(S[None, :, :], [tf.shape(y_pl)[0], 1, 1])
        # B, N
        post_mean = tf.linalg.matmul(XOXT, tf.linalg.lstsq(S, y_pl[:, :, None], fast=False))[:, :, 0]
        # B
        R = tf.linalg.matmul(OXT, tf.linalg.lstsq(S, tf.transpose(X_ext, (0,2,1)), fast=False))
        resolution = tf.linalg.trace(R)/tf.cast(tf.shape(R)[-1], R.dtype)
        if return_resolution:
            return sess.run([post_mean, resolution], {y_pl: Y, Sigma_pl: Sigma, cov_pl:cov_prior})
        return sess.run(post_mean, {y_pl: Y, Sigma_pl: Sigma})


def get_cov_matrix(freqs, x, deg, tec_scale, with_bias=True, bias_scale=np.pi):
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
    Nf = freqs.size
    N = 1000
    B = len(tec_scale)
    #N,B
    tec = tec_scale * np.random.normal(size=[N,B])
    #Nf, N, B
    phase = tec_conv[:,None, None] * tec
    if with_bias:
        const = np.random.uniform(-bias_scale, bias_scale, size=N)
        phase += const[None, :, None]

    Yreal = np.cos(phase).reshape((Nf, N*B))
    Yimag = np.sin(phase).reshape((Nf, N*B))
    # K, N, B
    preal = np.polyfit(x, Yreal, deg=deg).reshape((-1, N, B))
    pimag = np.polyfit(x, Yimag, deg=deg).reshape((-1, N, B))
    #K, N, B
    preal -= np.mean(preal, axis=1, keepdims=True)
    pimag -= np.mean(pimag, axis=1, keepdims=True)
    #K, K, B
    cov_real = np.mean(preal[:, None, :, :]*preal[None, :, :, :], axis=-2)
    cov_imag = np.mean(pimag[:, None, :, :]*pimag[None, :, :, :], axis=-2)
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

def smooth_gains(Yreal, Yimag, deg=2):
    """

    :param Yreal: [Npol, Nd, Na, Nf, Nt]
    :param Yimag: [Npol, Nd, Na, Nf, Nt]
    :param deg:
    :return:
    """
    Npol, Nd, Na, Nf, Nt = Yreal.shape



    phase = np.arctan2(Yimag, Yreal)
    Nf, N = Yreal.shape
    x = np.linspace(-1., 1., Nf)
    cov_real, cov_imag = get_cov_matrix(x, deg, tec_scale=200., with_bias=True, bias_scale=np.pi)

    _Yreal, _Yimag = np.copy(Yreal), np.copy(Yimag)

    _freqs = np.linspace(-1., 1., Nf)
    real = sum([median_filter(p, filter_size) * _freqs[:, None] ** (deg - i) for i, p in
                enumerate(np.polyfit(_freqs, _Yreal, deg=deg))])
    imag = sum([median_filter(p, filter_size) * _freqs[:, None] ** (deg - i) for i, p in
                enumerate(np.polyfit(_freqs, _Yimag, deg=deg))])

    res_real = np.abs(Yreal - real)
    flag = res_real > np.sort(res_real, axis=0)[-3]
    _Yreal[flag] = real[flag]
    res_imag = np.abs(Yimag - imag)
    flag = res_imag > np.sort(res_imag, axis=0)[-3]
    _Yimag[flag] = imag[flag]

    real = sum([median_filter(p, filter_size) * _freqs[:, None] ** (deg - i) for i, p in
                enumerate(np.polyfit(_freqs, _Yreal, deg=deg))])
    imag = sum([median_filter(p, filter_size) * _freqs[:, None] ** (deg - i) for i, p in
                enumerate(np.polyfit(_freqs, _Yimag, deg=deg))])

    _Yreal, _Yimag = np.copy(Yreal), np.copy(Yimag)

    res_real = np.abs(Yreal - real)
    flag = res_real > np.sort(res_real, axis=0)[-3]
    _Yreal[flag] = real[flag]
    res_imag = np.abs(Yimag - imag)
    flag = res_imag > np.sort(res_imag, axis=0)[-3]
    _Yimag[flag] = imag[flag]

    real = sum([median_filter(p, filter_size) * _freqs[:, None] ** (deg - i) for i, p in
                enumerate(np.polyfit(_freqs, _Yreal, deg=deg))])
    imag = sum([median_filter(p, filter_size) * _freqs[:, None] ** (deg - i) for i, p in
                enumerate(np.polyfit(_freqs, _Yimag, deg=deg))])

    return real, imag


def main(data_dir, working_dir, obs_num, block_size):
    working_dir = os.path.abspath(working_dir)
    os.makedirs(working_dir,exist_ok=True)
    data_dir = os.path.abspath(data_dir)
    logging.info("Changed dir to {}".format(working_dir))
    os.chdir(working_dir)
    sol_name='DDS4_full'
    datapack = os.path.join(data_dir, 'L{}_{}_merged.h5'.format(obs_num, sol_name))
    if not os.path.isfile(datapack):
        raise IOError("datapack doesn't exists {}".format(datapack))

    logging.info("Using working directory: {}".format(working_dir))
    select=dict(pol=slice(0,1,1))

    datapack = DataPack(datapack, readonly=False)
    logging.info("Creating smoothed/phase000+amplitude000")
    make_soltab(datapack, from_solset='sol000', to_solset='smoothed000', from_soltab='phase000',
                to_soltab=['phase000', 'amplitude000'])
    logging.info("Getting phase and amplitude data")
    datapack.current_solset = 'sol000'
    datapack.select(**select)
    axes = datapack.axes_phase
    antenna_labels, antennas = datapack.get_antennas(axes['ant'])
    patch_names, directions = datapack.get_directions(axes['dir'])
    timestamps, times = datapack.get_times(axes['time'])
    freq_labels, freqs = datapack.get_freqs(axes['freq'])
    pol_labels, pols = datapack.get_pols(axes['pol'])

    ref_dist = np.linalg.norm(antennas - antennas[0, :], axis=1)

    ref_ang_dist = great_circle_sep(directions.ra.rad, directions.dec.rad, directions.ra.rad[0], directions.dec.rad[0])

    Npol, Nd, Na, Nf, Nt = len(pols), len(directions), len(antennas), len(freqs), len(times)
    phase_raw, axes = datapack.phase
    amp_raw, axes = datapack.amplitude
    logging.info("Smoothing amplitudes in time and frequency.")
    amp_smooth = smoothamps(amp_raw)
    logging.info("Constructing gains.")
    # Npol, Nd, Na, Nf, Nt
    phase_raw = np.unwrap(phase_raw, axis=-1)
    c_phase_raw = phase_raw - np.mean(phase_raw, axis=-1, keepdims=True)
    Yimag_full = amp_smooth * np.sin(phase_raw)
    Yreal_full = amp_smooth * np.cos(phase_raw)
    x = np.linspace(-1.,1., Nf)
    deg = 2
    for start in range(0,Nt, block_size):
        stop = min(start+block_size, Nt)
        phase_var = np.mean(np.square(c_phase_raw[..., start:stop]), axis=-1)
        tec_scale = np.mean(phase_var*freqs[:, None]/8.4479745e6, axis=-1)
        cov_real, cov_imag = get_cov_matrix(freqs,x,deg, tec_scale.flatten(), True, 1.)
        Sigma = 0.1**2 * np.eye(Nf)
        Yreal_smooth, R = smooth(x, Yreal_full[..., start:stop], deg, cov_real.T, Sigma, return_resolution=True)
        Sigma = np.maximum(0.01**2, np.sqaure(Yreal_full[..., start:stop] - Yreal_smooth))


    # Nf, Npol*Nd*Na*Nt
    Yimag_full = Yimag_full.transpose((3,0,1,2,4)).reshape((Nf, -1))
    Yreal_full = Yreal_full.transpose((3,0,1,2,4)).reshape((Nf, -1))
    logging.info("Smoothing gains.")

    Yreal_full, Yimag_full = smooth_gains(Yreal_full, Yimag_full, 1, 2)
    # Npol, Nd, Na, Nf, Nt
    Yreal_full = Yreal_full.reshape((Nf, Npol, Nd, Na, Nt)).transpose((1,2,3,0,4))
    Yimag_full = Yimag_full.reshape((Nf, Npol, Nd, Na, Nt)).transpose((1,2,3,0,4))

    phase_smooth = np.arctan2(Yimag_full, Yreal_full)
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
            plt.imshow(diff_phase[0,d,a,:,:], origin='lower',aspect='auto',vmin=-0.1, vmax=0.1,cmap='coolwarm')
            plt.colorbar()
            plt.xlabel('time')
            plt.ylabel('freq')
            plt.savefig(os.path.join(working_dir, 'phase_diff_dir{:02d}_ant{:02d}.png'.format(d,a)))
            plt.close('all')
            plt.imshow(diff_amp[0, d, a, :, :], origin='lower', cmap='coolwarm', aspect='auto',vmin=np.percentile(diff_amp[0, d, a, :, :], 5), vmax=np.percentile(diff_amp[0, d, a, :, :], 95))
            plt.xlabel('time')
            plt.ylabel('freq')
            plt.colorbar()
            plt.savefig(os.path.join(working_dir, 'amp_diff_dir{:02d}_ant{:02d}.png'.format(d, a)))
            plt.close('all')
            for t in worst_times:
                plt.scatter(freqs, phase_raw[0,d,a,:,t])
                plt.plot(freqs, phase_smooth[0,d,a,:,t])
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
    parser.add_argument('--obs_num', help='Obs number L*',
                        default=None, type=int, required=True)
    parser.add_argument('--data_dir', help='Where are the ms files are stored.',
                        default=None, type=str, required=True)
    parser.add_argument('--working_dir', help='Where to perform the imaging.',
                        default=None, type=str, required=True)
    parser.add_argument('--block_size', help='Blocks over which to get stats for prior.',
                        default=None, type=str, required=True)

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