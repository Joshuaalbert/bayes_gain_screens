import os
os.environ['OMP_NUM_THREADS'] = "1"
import matplotlib
matplotlib.use('Agg')
import numpy as np
from scipy.ndimage import median_filter
from bayes_gain_screens import logging
from bayes_gain_screens.datapack import DataPack
from bayes_gain_screens.misc import make_soltab
import pylab as plt
import argparse

"""
This script is still being debugged/tested. 
Get's TEC from gains.
"""

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


def smooth_gains(Yreal, Yimag, filter_size=1, deg=2):
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


def main(data_dir, working_dir, obs_num):
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
    worst_ants = np.argsort(np.square(diff_phase).mean(0).mean(0).mean(-1).mean(-1))[-5:]
    worst_dirs = np.argsort(np.square(diff_phase).mean(0).mean(-1).mean(-1).mean(-1))[-5:]
    worst_times = np.argsort(np.square(diff_phase).mean(0).mean(0).mean(0).mean(0))[-5:]

    for d in worst_dirs:
        for a in worst_ants:
            plt.imshow(diff_phase[0,d,a,:,:], origin='lower',aspect='auto',vmin=-0.1, vmax=0.1,cmap='coolwarm')
            plt.xlabel('time')
            plt.ylabel('freq')
            plt.colorbar()
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