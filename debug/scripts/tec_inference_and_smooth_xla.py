import os
import pylab as plt
from scipy.ndimage import median_filter
from bayes_gain_screens import logging
from bayes_gain_screens.datapack import DataPack
from bayes_gain_screens.misc import make_soltab, great_circle_sep, apply_rolling_func_strided
from bayes_gain_screens.plotting import animate_datapack
from bayes_gain_screens.nlds_smoother import NLDSSmoother
from bayes_gain_screens.updates.gains_to_tec_with_amps_update import UpdateGainsToTecAmps
from bayes_gain_screens import TEC_CONV
from dask.multiprocessing import get
import argparse
from timeit import default_timer
import tensorflow.compat.v1 as tf
import networkx as nx
import sys

# this before JAX
from bayes_gain_screens.variational_hmm import NonLinearDynamicsSmoother, TecOnlyAmpDiagLinearPhaseDiagSigma
from jax import jit, soft_pmap
import jax.numpy as jnp
import numpy as onp
from jax.scipy.signal import _convolve_nd

"""
This script is still being debugged/tested. 
Get's TEC from gains.
"""

def link_overwrite(src, dst):
    if os.path.islink(dst):
        print("Unlinking pre-existing sym link {}".format(dst))
        os.unlink(dst)
    print("Linking {} -> {}".format(src, dst))
    os.symlink(src, dst)

def wrap(p):
    return jnp.arctan2(jnp.sin(p), jnp.cos(p))

def smoothamps(amps):
    freqkernel = 3
    timekernel = 61
    logging.info("Smoothing amplitudes with a median filter ({:.2f}MHz, {:.2f}minutes)".format(freqkernel*1.95, timekernel*0.5))
    amps = onp.where(amps < 0.5, 0.5, amps)
    amps = onp.where(amps > 2., 2., amps)
    ampssmoothed = onp.exp((median_filter(onp.log(amps), size=(1, 1, 1, freqkernel, timekernel), mode='reflect')))
    return jnp.array(ampssmoothed)

def windowed_mean(a, w, mode='reflect'):
    dims = len(a.shape)
    a = a
    kernel = jnp.reshape(jnp.ones(w)/w, [w]+[1]*(dims-1))
    _w1 = (w-1)//2
    _w2 = _w1 if (w%2 == 1) else _w1 + 1
    pad_width = [(_w1, _w2)] + [(0,0)]*(dims-1)
    a = jnp.pad(a, pad_width=pad_width, mode=mode)
    return _convolve_nd(a,kernel, mode='valid', precision=None)

def polyfit(x, y, deg):
    """
    x : array_like, shape (M,)
        x-coordinates of the M sample points ``(x[i], y[i])``.
    y : array_like, shape (M,) or (M, K)
        y-coordinates of the sample points. Several data sets of sample
        points sharing the same x-coordinates can be fitted at once by
        passing in a 2D-array that contains one dataset per column.
    deg : int
        Degree of the fitting polynomial
    Returns
    -------
    p : ndarray, shape (deg + 1,) or (deg + 1, K)
        Polynomial coefficients, highest power first.  If `y` was 2-D, the
        coefficients for `k`-th data set are in ``p[:,k]``.
    """
    order = int(deg) + 1
    if deg < 0:
        raise ValueError("expected deg >= 0")
    if x.ndim != 1:
        raise TypeError("expected 1D vector for x")
    if x.size == 0:
        raise TypeError("expected non-empty vector for x")
    if y.ndim < 1 or y.ndim > 2:
        raise TypeError("expected 1D or 2D array for y")
    if x.shape[0] != y.shape[0]:
        raise TypeError("expected x and y to have same length")
    rcond = len(x)*jnp.finfo(x.dtype).eps
    lhs = jnp.stack([x**(deg - i) for i in range(order)], axis=1)
    rhs = y
    scale = jnp.sqrt(jnp.sum(lhs * lhs, axis=0))
    lhs /= scale
    c, resids, rank, s = jnp.linalg.lstsq(lhs, rhs, rcond)
    c = (c.T / scale).T  # broadcast scale coefficients
    return c

def main(data_dir, working_dir, obs_num, ref_dir, ncpu, walking_reference):
    os.chdir(working_dir)
    logging.info("Performing TEC and constant variational inference.")
    dds4_h5parm = os.path.join(data_dir, 'L{}_DDS4_full_merged.h5'.format(obs_num))
    dds5_h5parm = os.path.join(working_dir, 'L{}_DDS5_full_merged.h5'.format(obs_num))
    linked_dds5_h5parm = os.path.join(data_dir, 'L{}_DDS5_full_merged.h5'.format(obs_num))
    link_overwrite(dds5_h5parm, linked_dds5_h5parm)

    logging.info("Looking for {}".format(dds4_h5parm))
    select = dict(pol=slice(0, 1, 1))
    dds4_datapack = DataPack(dds4_h5parm, readonly=False)
    logging.info("Creating smoothed/phase000+amplitude000")
    make_soltab(dds4_datapack, from_solset='sol000', to_solset='smoothed000', from_soltab='phase000',
                to_soltab=['phase000', 'amplitude000'], remake_solset=True, to_datapack=dds5_h5parm)
    logging.info("Creating directionally_referenced/tec000+const000")
    make_soltab(dds4_datapack, from_solset='sol000', to_solset='directionally_referenced', from_soltab='phase000',
                to_soltab=['tec000', 'const000'], remake_solset=True, to_datapack=dds5_h5parm)


    logging.info("Getting raw phases")
    dds4_datapack.current_solset = 'sol000'
    dds4_datapack.select(**select)
    axes = dds4_datapack.axes_phase
    antenna_labels, antennas = dds4_datapack.get_antennas(axes['ant'])
    patch_names, directions = dds4_datapack.get_directions(axes['dir'])
    radec = jnp.stack([directions.ra.rad, directions.dec.rad], axis=1)
    timestamps, times = dds4_datapack.get_times(axes['time'])
    freq_labels, freqs = dds4_datapack.get_freqs(axes['freq'])
    pol_labels, pols = dds4_datapack.get_pols(axes['pol'])
    Npol, Nd, Na, Nf, Nt = len(pols), len(directions), len(antennas), len(freqs), len(times)
    phase_raw, axes = dds4_datapack.phase
    amp_raw, axes = dds4_datapack.amplitude
    amp_smooth = smoothamps(amp_raw)

    amp_smooth, eff_const, phase_model, phase_raw, res_imag, res_real, results = inference(Na, Nd, Nf, Npol, Nt,
                                                                                           amp_raw, amp_smooth, freqs,
                                                                                           phase_raw)

    logging.info("Storing smoothed phase and amplitudes")
    dds5_datapack = DataPack(dds5_h5parm)
    dds5_datapack.current_solset = 'smoothed000'
    dds5_datapack.select(**select)
    dds5_datapack.phase = phase_model
    dds5_datapack.amplitude = amp_smooth

    logging.info("Storing TEC and const")
    dds5_datapack.current_solset = 'directionally_referenced'
    # Npol, Nd, Na, Nf, Nt
    dds5_datapack.select(**select)
    dds5_datapack.tec = results.post_mu[:,0]
    dds5_datapack.weights_tec = jnp.sqrt(results.post_Gamma[:,0,0])
    dds5_datapack.const = eff_const
    logging.info("Storing HMM params")
    onp.save(os.path.join(working_dir, "Sigma.npy"), results.Sigma)
    onp.save(os.path.join(working_dir, "Omega.npy"), results.Omega)
    logging.info("Done ddtec VI.")

    animate_datapack(dds5_h5parm, os.path.join(working_dir, 'tec_plots'), num_processes=ncpu,
                     solset='directionally_referenced',
                     observable='tec', vmin=-60., vmax=60., plot_facet_idx=True,
                     labels_in_radec=True, plot_crosses=False, phase_wrap=False,
                     flag_outliers=False)

    animate_datapack(dds5_h5parm, os.path.join(working_dir, 'const_plots'), num_processes=ncpu,
                     solset='directionally_referenced',
                     observable='const', vmin=-onp.pi, vmax=onp.pi, plot_facet_idx=True,
                     labels_in_radec=True, plot_crosses=False, phase_wrap=False,
                     flag_outliers=False)

    animate_datapack(dds5_h5parm, os.path.join(working_dir, 'tec_uncert_plots'), num_processes=ncpu,
                     solset='directionally_referenced',
                     observable='weights_tec', vmin=0., vmax=5., plot_facet_idx=True,
                     labels_in_radec=True, plot_crosses=False, phase_wrap=False,
                     flag_outliers=False)

    animate_datapack(dds5_h5parm, os.path.join(working_dir, 'smoothed_amp_plots'), num_processes=ncpu,
                     solset='smoothed000',
                     observable='amplitude', vmin=0.5, vmax=2., plot_facet_idx=True,
                     labels_in_radec=True, plot_crosses=False, phase_wrap=False)

    plot_results(Na, Nd, antenna_labels, working_dir, phase_model, phase_raw,
                 res_imag, res_real, results.post_mu[:,0].reshape((Npol,Nd,Na,Nt)),
                 results.Sigma.reshape((Npol,Nd,Na,Nt,2*Nf, 2*Nf)), results.Omega.reshape((Npol,Nd,Na,Nt,1,1)))

@jit
def inference(Na, Nd, Nf, Npol, Nt, amp_raw, amp_smooth, freqs, phase_raw):
    # Npol* Nd* Na, Nt, Nf
    phase_raw = phase_raw.transpose((0, 1, 2, 4, 3)).reshape((Npol * Nd * Na, Nt, Nf))
    amp_smooth = amp_smooth.transpose((0, 1, 2, 4, 3)).reshape((Npol * Nd * Na, Nt, Nf))
    tec_conv = TEC_CONV / freqs
    # Npol* Nd* Na, Nt, 2Nf
    Y_obs = jnp.concatenate([amp_raw * jnp.cos(phase_raw), amp_raw * jnp.sin(phase_raw)], axis=-1)
    Sigma = 0.5 * jnp.eye(2 * Nf)
    Omega = 5. * jnp.eye(1)
    mu0 = jnp.zeros(1)
    Gamma0 = 100. ** 2 * jnp.eye(1)
    hmm = NonLinearDynamicsSmoother(TecOnlyAmpDiagLinearPhaseDiagSigma(freqs))

    def constant_folded(Sigma, mu0, Gamma0, Omega,
                        tol, maxiter, omega_window, sigma_window, momentum):
        def f(Y_obs, amp):
            return hmm(Y_obs, Sigma, mu0, Gamma0, Omega, amp,
                       tol=tol, maxiter=maxiter, momentum=momentum,
                       omega_window=omega_window, sigma_window=sigma_window)

        return f

    pfun = soft_pmap(constant_folded(Sigma, mu0, Gamma0, Omega,
                                     tol=1., maxiter=10, omega_window=10, sigma_window=10,
                                     momentum=0.1),
                     in_axes=(0, 0))
    results = pfun(Y_obs, amp_smooth)
    post_mean_tec = results.post_mu[:, :, 0].reshape((Npol, Nd, Na, Nt))
    phase_model = post_mean_tec[..., None, :] * tec_conv[:, None]
    diff_phase = wrap(phase_raw - wrap(phase_model)).transpose((0, 1, 2, 4, 3)).reshape((Npol * Nd * Na * Nt, Nf))
    # N
    eff_phase_residual = polyfit(freqs / 1e6, diff_phase.T, deg=1)[0, :] * (freqs[-1] - freqs[0]) / 1e6 / 2.
    eff_phase_residual = windowed_mean(eff_phase_residual.reshape((Npol * Nd * Na, Nt)).T, 91).T
    eff_const = jnp.reshape(eff_phase_residual / 0.157, (Npol, Nd, Na, Nt))
    phase_raw = phase_raw - eff_const[..., None, :]
    Y_obs = jnp.concatenate([amp_raw * jnp.cos(phase_raw), amp_raw * jnp.sin(phase_raw)], axis=-1)
    results = pfun(Y_obs, amp_smooth)
    post_mean_tec = results.post_mu[:, :, 0].reshape((Npol, Nd, Na, Nt))
    phase_model = post_mean_tec[..., None, :] * tec_conv[:, None] + eff_const[..., None, :]
    res_real = amp_raw * jnp.cos(phase_raw) - amp_smooth * jnp.cos(phase_model)
    res_imag = amp_raw * jnp.sin(phase_raw) - amp_smooth * jnp.sin(phase_model)
    return amp_smooth, eff_const, phase_model, phase_raw, res_imag, res_real, results


def plot_results(Na, Nd, antenna_labels, working_dir, phase_model,
                 phase_raw, res_imag, res_real, tec_mean_array, Sigma_array, Omega_array):
    logging.info("Plotting results.")
    summary_dir = os.path.join(working_dir, 'summaries')
    os.makedirs(summary_dir, exist_ok=True)
    for i in range(Na):
        for j in range(Nd):
            slice_phase_data = wrap(phase_raw[0, j, i, :, :])
            slice_phase_model = wrap(phase_model[0, j, i, :, :])
            slice_res_real = res_real[0, j, i, :, :]
            slice_res_imag = res_imag[0, j, i, :, :]
            time_array = onp.arange(slice_res_real.shape[-1])
            colors = plt.cm.jet(onp.arange(slice_res_real.shape[-1]) / slice_res_real.shape[-1])
            # Nf, Nt
            _slice_res_real = slice_res_real - onp.mean(slice_res_real, axis=0)
            _slice_res_imag = slice_res_imag - onp.mean(slice_res_imag, axis=0)
            slice_tec = tec_mean_array[0, j, i, :]
            fig, axs = plt.subplots(2, 2, figsize=(15, 15))
            diff_phase = wrap(wrap(slice_phase_data) - wrap(slice_phase_model))
            for nu in range(slice_res_real.shape[-2]):
                f_c = plt.cm.binary((nu + 1) / slice_res_real.shape[-2])
                colors_ = (f_c + colors) / 2. * onp.array([1., 1., 1., 1. - (nu + 1) / slice_res_real.shape[-2]])
                axs[0][0].scatter(time_array, onp.abs(slice_res_real[nu, :]), c=colors_, marker='.')
                axs[0][0].scatter(time_array, -onp.abs(slice_res_imag[nu, :]), c=colors_, marker='.')
            axs[0][0].set_title("Real and Imag residuals")
            axs[0][0].hlines(0., time_array.min(), time_array.max())
            vmin, vmax = onp.percentile(diff_phase, [5,95])
            axs[0][1].imshow(diff_phase, origin='lower', aspect='auto', cmap='coolwarm', vmin=vmin,
                             vmax=vmax)
            axs[0][1].set_title('Phase residuals [{:.1f},{:.1f}]'.format(vmin,vmax))
            axs[0][1].set_xlabel('Time')
            axs[0][1].set_ylabel('Freq')
            axs[1][0].scatter(time_array, slice_tec, c=colors)
            axs[1][0].set_title("TEC")
            axs[1][0].set_xlabel('Time')
            axs[1][0].set_ylabel('TEC [mTECU]')
            #Nt, Nf
            _sigma = onp.sqrt(onp.diagonal(Sigma_array[0,j,i,:,:,:], axis1=-2, axis2=-1))
            _sigma_mean = onp.mean(_sigma, axis=-1)
            _sigma_std = onp.std(_sigma, axis=-1)
            _omega = onp.sqrt(Omega_array[0,j,i,:,0,0])
            # _omega_std = np.exp(np.std(0.5 * np.log(_omega), axis=-1))
            _omega_mean = _omega

            _t = onp.arange(len(_sigma))
            axs[1][1].plot(_t, _sigma_mean, label='sigma')
            axs[1][1].fill_between(_t, _sigma_mean - _sigma_std, _sigma_mean + _sigma_std, alpha=0.5)
            omega_axis = axs[1][1].twinx()
            omega_axis.plot(_t, _omega_mean, color='red', label='omega')
            axs[1][1].set_title("sigma and omega")
            axs[1][1].set_ylabel("mean data noise [1]")
            omega_axis.set_ylabel("omega DDTEC [mTECU]")
            axs[1][1].legend()
            omega_axis.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(summary_dir, 'summary_{}_dir{:02d}.png'.format(antenna_labels[i].decode(), j)))
            plt.close('all')


def add_args(parser):
    parser.register("type", "bool", lambda v: v.lower() == "true")
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
    parser.add_argument('--walking_reference', help='Whether to remove bias by rereferencing in a minimum distance spanning tree walk.',
                        default=False, type="bool", required=False)

def test_main():
    main(data_dir='/home/albert/store/root_dense/L667218/download_archive',
         working_dir='/home/albert/store/root_dense/L667218/tec_inference_and_smooth_test',
         obs_num=667218,
         ncpu=32,
         ref_dir=0,
         walking_reference=False
         )


if __name__ == '__main__':
    if len(sys.argv) == 1:
        test_main()
        exit(0)
    parser = argparse.ArgumentParser(
        description='Variational inference of DDTEC and a constant term. Updates the smoothed000 solset too.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_args(parser)
    flags, unparsed = parser.parse_known_args()
    print("Running with:")
    for option, value in vars(flags).items():
        print("    {} -> {}".format(option, value))
    main(**vars(flags))
