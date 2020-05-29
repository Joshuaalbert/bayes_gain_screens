import os
os.environ['OMP_NUM_THREADS'] = "1"
os.environ['NUMEXPR_MAX_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
import numpy as np
from scipy.optimize import minimize
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

def stack_complex(y):
    """
    Stacks complex real and imaginary parts
    :param y: [..., N]
    :return: [...,2N]
    real on top of imag
    """
    return np.concatenate([y.real, y.imag], axis=-1)


def loss(params, y_real, y_imag, sigma_real, sigma_imag,tec_conv, clock_conv, amp):
    tec = params[0, None]
    clock = params[1, None]
    const = params[2, None]

    phase = tec * tec_conv + const + clock * clock_conv
    return np.sum(np.square((amp*np.cos(phase) - y_real) / sigma_real) + np.square((amp*np.sin(phase) - y_imag) / sigma_imag))

def sequential_solve(amps, Yreal_data_dd, Yimag_data_dd, Yreal_data_di, Yimag_data_di, freqs, working_dir, smooth_window, debug=False):
    """
    Run on blocks of time.

    :param Yreal:
        [D, Nf, Nt]
    :param Yimag:
    :param freqs:
    :return:
        [D, Nt], [D, Nt]
    """
    if debug:
        debug_dir = os.path.join(working_dir, 'phase_residuals')
        os.makedirs(debug_dir, exist_ok=True)

    D, Nf, N = Yreal_data_dd.shape

    tec_conv = TEC_CONV / freqs
    clock_conv = 2. * np.pi * 1e-9 * freqs

    smoothed_phase_array = np.zeros((D, Nf, N))
    tec_mean_array = np.zeros((D, N))
    tec_uncert_array = np.zeros((D, N))
    const_array = np.zeros((D, N))
    Sigma_array = np.zeros((D, N, 2*Nf, 2*Nf))
    Omega_array = np.zeros((D, N, 1, 1))

    update = UpdateGainsToTecAmps(freqs, S=100, tec_scale=300., force_diag_Sigma=True, force_diag_Omega=True,
                                        Sigma_window=1, Omega_window=31)

    config = tf.ConfigProto(intra_op_parallelism_threads=1,
                            inter_op_parallelism_threads=1,
                            allow_soft_placement=True,
                            device_count={'CPU': 1})
    for d in range(D):
        t0 = default_timer()

        mu_0 = np.array([0., 0.])
        Gamma_0 = np.diag([200., 2 * np.pi]) ** 2

        Sigma_0 = 0.5 ** 2 * np.eye(2 * Nf)
        Omega_0 = np.diag([100., 0.01]) ** 2

        Y = np.transpose(Yreal_data_dd[d, :, :] + 1j * Yimag_data_dd[d, :, :])
        # each timestep get its own data uncertainty, and no coupling
        res = NLDSSmoother(2, 2 * Nf, update=update, momentum=0., serve_shapes=[[Nf]],
                           session=tf.Session(graph=tf.Graph(), config=config),
                           freeze_Omega=True).run(
            stack_complex(Y), Sigma_0, Omega_0, mu_0, Gamma_0, 1, serve_values=[amps[d, :, :].T])

        Sigma_0 = res['Sigma']#np.maximum(0.01 ** 2, np.mean(res['Sigma'], axis=0))
        Omega_0 = res['Omega']#np.maximum(0.1 ** 2, np.mean(res['Omega'], axis=0))
        mu_0 = res['mu0']
        Gamma_0 = res['Gamma0']
        # freeze sigma now and couple
        res = NLDSSmoother(2, 2 * Nf, update=update, momentum=0., serve_shapes=[[Nf]],
                           session=tf.Session(graph=tf.Graph(), config=config),
                           freeze_Sigma=True).run(
            stack_complex(Y), Sigma_0, Omega_0, mu_0, Gamma_0, 1, serve_values=[amps[d, :, :].T])



        ###
        # BEGIN subtract constant approximation
        #N, Nf
        phase_data = np.angle(Y)
        amp_data = np.abs(Y)
        #N, Nf
        diff_phase = wrap(phase_data - wrap(res['post_mu'][:,0:1] * tec_conv))
        #N
        eff_phase_residual = np.polyfit(freqs/1e6, diff_phase.T, deg=1)[0, :]*(freqs[-1] - freqs[0])/1e6/2.
        eff_const = eff_phase_residual / 0.157
        #N
        eff_const = median_filter(eff_const, size=(smooth_window,))
        const_array[d,:] = eff_const
        #N, Nf
        Y_mod = amp_data*np.exp(1j*(phase_data - eff_const[:, None]))
        res = NLDSSmoother(2, 2 * Nf, update=update, momentum=0., serve_shapes=[[Nf]],
                           session=tf.Session(graph=tf.Graph(), config=config),
                           freeze_Sigma=True, freeze_Omega=False).run(
            stack_complex(Y_mod), res['Sigma'], res['Omega'], res['mu0'], res['Gamma0'], 2, serve_values=[amps[d, :, :].T])
        ### END subtract constant approximation
        tec_mean_array[d, :] = res['post_mu'][:, 0]
        tec_uncert_array[d, :] = np.sqrt(res['post_Gamma'][:, 0, 0])
        Sigma_array[d, :, : , :] = res['Sigma']
        Omega_array[d, :, :, :] = res['Omega'][:, 0:1, 0:1]
        logging.info("DDTEC Levy uncert: {:.2f} +- {:.2f} mTECU".format(np.mean(np.sqrt(res['Omega'][:,0,0])), np.std(np.sqrt(res['Omega'][:,0,0]))))
        logging.info("Timing {:.2f} timesteps / second".format(N / (default_timer() - t0)))
        if debug:
            phase_model = tec_mean_array[d, None, :] * tec_conv[:, None]
            phase_diff = wrap(wrap(phase_model) - np.arctan2(Yimag_data_dd[d,:, :], Yreal_data_dd[d, :, :]))
            plt.imshow(phase_diff,origin='lower', vmin=-0.2, vmax= 0.2,
                       cmap='coolwarm', aspect='auto',
                       extent=(0, N, freqs.min(), freqs.max()))
            plt.xlabel('time')
            plt.ylabel('freq [Hz]')
            plt.savefig(os.path.join(debug_dir, 'phase_diff_{:04d}.png'.format(d)))
            plt.close('all')


        params = [minimize(loss, np.zeros(3), args=(Yreal_data_di[d,:,t], Yimag_data_di[d,:,t],
                                                    np.sqrt(np.diag(Sigma_array[d,t,:Nf,:Nf])+0.05**2),
                                                    np.sqrt(np.diag(Sigma_array[d,t,Nf:,Nf:])+0.05**2),
                                                    tec_conv, clock_conv, amps[d, :, t]),
                           method='BFGS').x for t in range(N)]
        #Nf, Nt
        smoothed_phase_array[d,:,:] = np.stack([p[0]*tec_conv + p[1]*clock_conv + p[2] for p in params], axis=1)
        #Nf, Nt
        res_real = Yreal_data_di[d,:,:] - amps[d, :, :]*np.cos(smoothed_phase_array[d,:,:])
        #Nf, Nt
        sigma_real = np.sqrt(np.square(res_real)*0.5 + 0.5*apply_rolling_func_strided(lambda x: np.mean(x, axis=-1), np.square(res_real), 5, piecewise_constant=False))
        sigma_real = np.maximum(0.01,sigma_real)
        # sigma_real = np.where(np.abs(res_real) > 2. * sigma_real, 5. * sigma_real, sigma_real)

        res_imag = Yimag_data_di[d, :, :] - amps[d, :, :]*np.sin(smoothed_phase_array[d, :, :])
        sigma_imag = np.sqrt(np.square(res_imag)*0.5 + 0.5*apply_rolling_func_strided(lambda x: np.mean(x, axis=-1), np.square(res_imag), 5, piecewise_constant=False))
        sigma_imag = np.maximum(0.01,sigma_imag)
        # sigma_imag = np.where(np.abs(res_imag) > 2. * sigma_imag, 5. * sigma_imag, sigma_imag)

        params = [minimize(loss, params[t], args=(Yreal_data_di[d, :, t], Yimag_data_di[d, :, t],
                                                    sigma_real[:,t],
                                                    sigma_imag[:,t],
                                                    tec_conv,
                                                    clock_conv,
                                                    amps[d, :, t]),
                           method='BFGS').x for t in range(N)]
        smoothed_phase_array[d, :, :] = np.stack([p[0] * tec_conv + p[1] * clock_conv + p[2] for p in params], axis=1)
    return tec_mean_array, tec_uncert_array, Sigma_array, Omega_array, smoothed_phase_array, const_array

def wrap(p):
    return np.arctan2(np.sin(p), np.cos(p))

def smoothamps(amps):
    freqkernel = 3
    timekernel = 61
    logging.info("Smoothing amplitudes with a median filter ({:.2f}MHz, {:.2f}minutes)".format(freqkernel*1.95, timekernel*0.5))
    amps = np.where(amps < 0.5, 0.5, amps)
    amps = np.where(amps > 2., 2., amps)
    # idxh = np.where(amps > 2.)
    # idxl = np.where(amps < 0.25)
    # median = np.tile(np.nanmedian(amps, axis=-1, keepdims=True), (1, 1, 1, 1, amps.shape[-1]))
    # amps[idxh] = median[idxh]
    # amps[idxl] = median[idxl]
    ampssmoothed = np.exp((median_filter(np.log(amps), size=(1, 1, 1, freqkernel, timekernel), mode='reflect')))
    return ampssmoothed


def main(data_dir, working_dir, obs_num, ref_dir, ncpu, walking_reference, const_smooth_window):
    os.chdir(working_dir)
    logging.info("Performing TEC and constant variational inference.")
    dds4_h5parm = os.path.join(data_dir, 'L{}_DDS4_full_merged.h5'.format(obs_num))

    dds5_h5parm = os.path.join(working_dir, 'L{}_DDS5_full_merged.h5'.format(obs_num))
    linked_dds5_h5parm = os.path.join(data_dir, 'L{}_DDS5_full_merged.h5'.format(obs_num))

    logging.info("Looking for {}".format(dds4_h5parm))
    select = dict(pol=slice(0, 1, 1))
    dds4_datapack = DataPack(dds4_h5parm, readonly=False)
    logging.info("Creating smoothed/phase000+amplitude000")
    make_soltab(dds4_datapack, from_solset='sol000', to_solset='smoothed000', from_soltab='phase000',
                to_soltab=['phase000', 'amplitude000'], remake_solset=True, to_datapack=dds5_h5parm)
    logging.info("Creating directionally_referenced/tec000+const000")
    make_soltab(dds4_datapack, from_solset='sol000', to_solset='directionally_referenced', from_soltab='phase000',
                to_soltab=['tec000', 'const000'], remake_solset=True, to_datapack=dds5_h5parm)

    link_overwrite(dds5_h5parm, linked_dds5_h5parm)

    logging.info("Getting raw phases")
    dds4_datapack.current_solset = 'sol000'
    dds4_datapack.select(**select)
    axes = dds4_datapack.axes_phase
    antenna_labels, antennas = dds4_datapack.get_antennas(axes['ant'])
    patch_names, directions = dds4_datapack.get_directions(axes['dir'])
    radec = np.stack([directions.ra.rad, directions.dec.rad], axis=1)
    timestamps, times = dds4_datapack.get_times(axes['time'])
    freq_labels, freqs = dds4_datapack.get_freqs(axes['freq'])
    pol_labels, pols = dds4_datapack.get_pols(axes['pol'])
    Npol, Nd, Na, Nf, Nt = len(pols), len(directions), len(antennas), len(freqs), len(times)
    phase_raw, axes = dds4_datapack.phase
    amp_raw, axes = dds4_datapack.amplitude
    amp_smooth = smoothamps(amp_raw)

    tec_conv = TEC_CONV / freqs
    tec_mean_array = np.zeros((Npol, Nd, Na, Nt))
    tec_uncert_array = np.zeros((Npol, Nd, Na, Nt))
    const_array = np.zeros((Npol,Nd, Na, Nt))
    Sigma_array = np.zeros((Npol, Nd, Na, Nt, 2*Nf, 2*Nf))
    Omega_array = np.zeros((Npol, Nd, Na, Nt, 1, 1))
    smoothed_phase_array = np.zeros((Npol, Nd, Na, Nf, Nt))
    g = nx.complete_graph(radec.shape[0])
    for u, v in g.edges:
        g[u][v]['weight'] = great_circle_sep(*radec[u, :], *radec[v, :])
    h = nx.minimum_spanning_tree(g)
    # walk_order = [(0,0)]+list(nx.bfs_edges(h, 0))
    # walk_order = list(nx.bfs_edges(h, ref_dir))
    walk_order =  [(ref_dir, ref_dir)] + list(nx.bfs_edges(h, ref_dir))
    proc_idx = 0
    for (next_ref_dir, solve_dir) in walk_order:
        if not walking_reference:
            next_ref_dir = ref_dir
        logging.info("Solving dir: {}".format(solve_dir))
        phase_di = phase_raw[:, next_ref_dir:next_ref_dir+1, ...]
        logging.info("Referencing dir: {}".format(next_ref_dir))
        phase_dd = phase_raw[:, solve_dir:solve_dir+1, ...] - phase_di

        # Npol, 1, Na, Nf, Nt
        Yimag_data_dd = amp_raw[:, solve_dir:solve_dir + 1, ...] * np.sin(phase_dd)
        Yreal_data_dd = amp_raw[:, solve_dir:solve_dir + 1, ...] * np.cos(phase_dd)
        Yimag_data_dd = Yimag_data_dd.reshape((-1, Nf, Nt))
        Yreal_data_dd = Yreal_data_dd.reshape((-1, Nf, Nt))

        amps_mod = amp_smooth[:, solve_dir:solve_dir+1, ...].reshape((-1, Nf, Nt))

        Yimag_data_di = amp_raw[:, solve_dir:solve_dir + 1, ...] * np.sin(phase_raw[:, solve_dir:solve_dir + 1, ...])
        Yreal_data_di = amp_raw[:, solve_dir:solve_dir + 1, ...] * np.cos(phase_raw[:, solve_dir:solve_dir + 1, ...])
        Yimag_data_di = Yimag_data_di.reshape((-1, Nf, Nt))
        Yreal_data_di = Yreal_data_di.reshape((-1, Nf, Nt))

        logging.info("Building dask")
        D = Yreal_data_dd.shape[0]
        logging.info("Batch-size {}".format(D))
        num_processes = min(D, ncpu)
        logging.info("Using {} of {} cpus".format(num_processes, ncpu))
        dsk = {}
        keys = []
        for c,i in enumerate(range(0, D, D // num_processes)):
            start = i
            stop = min(i + (D // num_processes), D)
            dsk[str(c)] = (sequential_solve, amps_mod[start:stop, :, :],
                           Yreal_data_dd[start:stop, :, :], Yimag_data_dd[start:stop, :, :],
                           Yreal_data_di[start:stop, :, :], Yimag_data_di[start:stop, :, :],
                           freqs, os.path.join(working_dir,'proc_{:04d}'.format(proc_idx)),
                           const_smooth_window)
            proc_idx += 1
            keys.append(str(c))
        logging.info("Running dask on {} processes".format(num_processes))
        results = get(dsk, keys, num_workers=num_processes)
        logging.info("Finished dask.")
        tec_mean = np.zeros((D, Nt))
        tec_uncert = np.zeros((D, Nt))
        const = np.zeros((D, Nt))
        Sigma = np.zeros((D, Nt, 2*Nf, 2*Nf))
        Omega = np.zeros((D, Nt, 1, 1))
        smoothed_phase = np.zeros((D, Nf, Nt))

        for c, i in enumerate(range(0, D, D // num_processes)):
            start = i
            stop = min(i + (D // num_processes), D)
            tec_mean[start:stop, :] = results[c][0]
            tec_uncert[start:stop, :] = results[c][1]
            Sigma[start:stop,:, :, :] = results[c][2]
            Omega[start:stop,:, :, :] = results[c][3]
            smoothed_phase[start:stop, :, :] = results[c][4]
            const[start:stop, :] = results[c][5]

        tec_mean = tec_mean.reshape((Npol, 1, Na, Nt))
        tec_uncert = tec_uncert.reshape((Npol, 1, Na, Nt))
        const = const.reshape((Npol, 1, Na, Nt))
        Sigma = Sigma.reshape((Npol, 1, Na, Nt, 2*Nf, 2*Nf))
        Omega = Omega.reshape((Npol, 1, Na, Nt, 1, 1))
        smoothed_phase = smoothed_phase.reshape((Npol, 1, Na, Nf, Nt))
        if walking_reference:
            logging.info("Re-referencing to {}".format(ref_dir))
            # phase_smooth[:, solve_dir:solve_dir+1, ...] = tec_mean[..., None, :] * tec_conv[:, None] + phase_di
            #Reference to ref_dir 0: tau_ij + tau_jk = tau_ik
            tec_mean_array[:, solve_dir:solve_dir+1,...] = tec_mean + tec_mean_array[:,next_ref_dir:next_ref_dir+1,...]
            tec_uncert_array[:, solve_dir:solve_dir+1, ...] = np.sqrt(tec_uncert**2 + tec_uncert_array[:, next_ref_dir:next_ref_dir+1, ...]**2)
            const_array[:, solve_dir:solve_dir+1, ...] = const + const_array[:,next_ref_dir:next_ref_dir+1,...]
            Sigma_array[:, solve_dir:solve_dir+1, ...] = Sigma
            Omega_array[:, solve_dir:solve_dir+1, ...] = Omega
            smoothed_phase_array[:, solve_dir:solve_dir+1, ...] = smoothed_phase
        else:
            tec_mean_array[:, solve_dir:solve_dir + 1, ...] = tec_mean
            tec_uncert_array[:, solve_dir:solve_dir + 1, ...] = tec_uncert
            const_array[:, solve_dir:solve_dir + 1, ...] = const
            Sigma_array[:, solve_dir:solve_dir + 1, ...] = Sigma
            Omega_array[:, solve_dir:solve_dir + 1, ...] = Omega
            smoothed_phase_array[:, solve_dir:solve_dir + 1, ...] = smoothed_phase

    # phase_smooth_uncert = np.abs(tec_conv[:, None] * tec_uncert_array[..., None, :])
    phase_model = tec_mean_array[..., None, :]*tec_conv[:, None] + smoothed_phase_array[:, ref_dir:ref_dir+1, ...] + const_array[...,None,:]
    res_real = amp_raw * np.cos(phase_raw) - amp_smooth*np.cos(phase_model)
    res_imag = amp_raw * np.sin(phase_raw) - amp_smooth*np.sin(phase_model)

    logging.info("Storing smoothed phase and amplitudes")
    dds5_datapack = DataPack(dds5_h5parm)
    dds5_datapack.current_solset = 'smoothed000'
    dds5_datapack.select(**select)
    dds5_datapack.phase = smoothed_phase_array
    dds5_datapack.amplitude = amp_smooth

    logging.info("Storing TEC and const")
    dds5_datapack.current_solset = 'directionally_referenced'
    # Npol, Nd, Na, Nf, Nt
    dds5_datapack.select(**select)
    dds5_datapack.tec = tec_mean_array
    dds5_datapack.weights_tec = tec_uncert_array
    dds5_datapack.const = const_array
    logging.info("Storing HMM params")
    np.save(os.path.join(working_dir, "Sigma.npy"), Sigma_array)
    np.save(os.path.join(working_dir, "Omega.npy"), Omega_array)
    logging.info("Done ddtec VI.")

    animate_datapack(dds5_h5parm, os.path.join(working_dir, 'tec_plots'), num_processes=ncpu,
                     solset='directionally_referenced',
                     observable='tec', vmin=-60., vmax=60., plot_facet_idx=True,
                     labels_in_radec=True, plot_crosses=False, phase_wrap=False,
                     flag_outliers=False)

    animate_datapack(dds5_h5parm, os.path.join(working_dir, 'const_plots'), num_processes=ncpu,
                     solset='directionally_referenced',
                     observable='const', vmin=-np.pi, vmax=np.pi, plot_facet_idx=True,
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
                 res_imag, res_real, tec_mean_array,Sigma_array, Omega_array)


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
            time_array = np.arange(slice_res_real.shape[-1])
            colors = plt.cm.jet(np.arange(slice_res_real.shape[-1]) / slice_res_real.shape[-1])
            # Nf, Nt
            _slice_res_real = slice_res_real - np.mean(slice_res_real, axis=0)
            _slice_res_imag = slice_res_imag - np.mean(slice_res_imag, axis=0)
            slice_tec = tec_mean_array[0, j, i, :]
            fig, axs = plt.subplots(2, 2, figsize=(15, 15))
            diff_phase = wrap(wrap(slice_phase_data) - wrap(slice_phase_model))
            for nu in range(slice_res_real.shape[-2]):
                f_c = plt.cm.binary((nu + 1) / slice_res_real.shape[-2])
                colors_ = (f_c + colors) / 2. * np.array([1., 1., 1., 1. - (nu + 1) / slice_res_real.shape[-2]])
                axs[0][0].scatter(time_array, np.abs(slice_res_real[nu, :]), c=colors_, marker='.')
                axs[0][0].scatter(time_array, -np.abs(slice_res_imag[nu, :]), c=colors_, marker='.')
            axs[0][0].set_title("Real and Imag residuals")
            axs[0][0].hlines(0., time_array.min(), time_array.max())
            axs[0][1].imshow(diff_phase, origin='lower', aspect='auto', cmap='coolwarm', vmin=-0.2,
                             vmax=0.2)
            axs[0][1].set_title('Phase residuals [-0.2,0.2]')
            axs[0][1].set_xlabel('Time')
            axs[0][1].set_ylabel('Freq')
            axs[1][0].scatter(time_array, slice_tec, c=colors)
            axs[1][0].set_title("TEC")
            axs[1][0].set_xlabel('Time')
            axs[1][0].set_ylabel('TEC [mTECU]')
            #Nt, Nf
            _sigma = np.sqrt(np.diagonal(Sigma_array[0,j,i,:,:,:], axis1=-2, axis2=-1))
            _sigma_mean = np.mean(_sigma, axis=-1)
            _sigma_std = np.std(_sigma, axis=-1)
            _omega = np.sqrt(Omega_array[0,j,i,:,0,0])
            # _omega_std = np.exp(np.std(0.5 * np.log(_omega), axis=-1))
            _omega_mean = _omega

            _t = np.arange(len(_sigma))
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
    parser.add_argument('--const_smooth_window', help='How long the constant smooth window is.',
                        default=300, type=int, required=False)

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
