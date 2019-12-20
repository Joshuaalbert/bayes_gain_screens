import os
os.environ['OMP_NUM_THREADS'] = "1"
os.environ['NUMEXPR_MAX_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
import numpy as np
import pylab as plt
from scipy.ndimage import median_filter
from bayes_gain_screens import logging
from bayes_gain_screens.datapack import DataPack
from bayes_gain_screens.misc import make_soltab, great_circle_sep
from bayes_gain_screens.plotting import animate_datapack
from bayes_gain_screens.nlds_smoother import NLDSSmoother
from bayes_gain_screens.updates.gains_to_tec_update import UpdateGainsToTec
from bayes_gain_screens import TEC_CONV
from dask.multiprocessing import get
import argparse
from timeit import default_timer
import tensorflow as tf
import networkx as nx

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
    return np.concatenate([y.real, y.imag], axis=-1)

def sequential_solve(Yreal, Yimag, freqs, working_dir):
    """
    Run on blocks of time.

    :param Yreal:
        [D, Nf, Nt]
    :param Yimag:
    :param freqs:
    :return:
        [D, Nt], [D, Nt]
    """
    debug_dir = os.path.join(working_dir, 'phase_residuals')
    os.makedirs(debug_dir, exist_ok=True)

    D, Nf, N = Yreal.shape

    tec_mean_array = np.zeros((D, N))
    tec_uncert_array = np.zeros((D, N))
    obs_cov_array = np.zeros((D, 2*Nf, 2*Nf))
    update = UpdateGainsToTec(freqs, S=200, tec_scale=300., spacing=10., force_diag_Sigma=True)
    config = tf.ConfigProto(intra_op_parallelism_threads=1,
                            inter_op_parallelism_threads=1,
                            allow_soft_placement=True,
                            device_count={'CPU': 1})
    for d in range(D):
        t0 = default_timer()
        Sigma_0 = 1 ** 2 * np.eye(2 * Nf)
        Omega_0 = np.diag([50., 0.1]) ** 2
        mu_0 = np.array([0., 0.])
        Gamma_0 = np.diag([200., 2 * np.pi]) ** 2
        ###
        # warm-up
        # logging.info("On {}: Warming up".format(d))
        # B, Nf
        Y_warmup = np.transpose(Yreal[d, :, : 50] + 1j * Yimag[d, :, :50])
        res = NLDSSmoother(2, 2*Nf, N, update=update, momentum=0.9, session=tf.Session(graph=tf.Graph(), config=config)).run(stack_complex(Y_warmup), Sigma_0, Omega_0, mu_0,
                                                                      Gamma_0, 10)
        Sigma_0 = res['Sigma']
        Omega_0 = res['Omega']
        mu_0 = res['mu_0']
        Gamma_0 = res['Gamma_0']
        # logging.info("On {}: Full chain".format(d))
        Y = np.transpose(Yreal[d, :, :] + 1j * Yimag[d, :, :])
        res = NLDSSmoother(2, 2*Nf, N, update=update, momentum=0.1, session=tf.Session(graph=tf.Graph(), config=config)).run(stack_complex(Y), Sigma_0, Omega_0,
                                                                      mu_0,
                                                                      Gamma_0, 2)
        tec_mean_array[d, :] = res['post_mu'][:, 0]
        tec_uncert_array[d, :] = np.sqrt(res['post_Gamma'][:, 0, 0])
        obs_cov_array[d, :, :] = res['Sigma']
        logging.info("DDTEC Levy uncert: {:.2f} mTECU".format(np.sqrt(res['Omega'][0,0])))
        logging.info("Timing {:.2f} timesteps / second".format(N / (default_timer() - t0)))
        phase_model = tec_mean_array[d, None, :] * TEC_CONV/freqs[:, None]
        phase_diff = wrap(wrap(phase_model) - np.arctan2(Yimag[d,:, :], Yreal[d, :, :]))
        plt.imshow(phase_diff,origin='lower', vmin=-0.2, vmax= 0.2,
                   cmap='coolwarm', aspect='auto',
                   extent=(0, N, freqs.min(), freqs.max()))
        plt.xlabel('time')
        plt.ylabel('freq [Hz]')
        plt.savefig(os.path.join(debug_dir, 'phase_diff_{:04d}.png'.format(d)))
        plt.close('all')

        # plt.imshow(res['Sigma'], origin='lower',
        #            cmap='bone', aspect='auto')
        # plt.colorbar()
        # plt.savefig(os.path.join(debug_dir, 'obs_cov_{:04d}.png'.format(d)))
        #
        # plt.close('all')


    return tec_mean_array, tec_uncert_array, obs_cov_array

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


def main(data_dir, working_dir, obs_num, ref_dir, ncpu, walking_reference):
    os.chdir(working_dir)
    logging.info("Performing TEC and constant variational inference.")
    merged_h5parm = os.path.join(data_dir, 'L{}_DDS4_full_merged.h5'.format(obs_num))
    select = dict(pol=slice(0, 1, 1))
    datapack = DataPack(merged_h5parm, readonly=False)
    logging.info("Creating directionally_referenced/tec000+const000")
    make_soltab(datapack, from_solset='sol000', to_solset='directionally_referenced', from_soltab='phase000',
                to_soltab=['tec000'])
    logging.info("Getting raw phases")
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
    logging.info("Getting smooth phase and amplitude data")
    datapack.current_solset = 'smoothed000'
    datapack.select(**select)
    phase_smooth, axes = datapack.phase
    amp_smooth, axes = datapack.amplitude

    tec_conv = TEC_CONV / freqs
    tec_mean_array = np.zeros((Npol, Nd, Na, Nt))
    tec_uncert_array = np.zeros((Npol, Nd, Na, Nt))
    obs_cov_array = np.zeros((Npol, Nd, Na, 2*Nf, 2*Nf))
    g = nx.complete_graph(radec.shape[0])
    for u, v in g.edges:
        g[u][v]['weight'] = great_circle_sep(*radec[u, :], *radec[v, :])
    h = nx.minimum_spanning_tree(g)
    # walk_order = [(0,0)]+list(nx.bfs_edges(h, 0))
    walk_order = list(nx.bfs_edges(h, ref_dir))
    proc_idx = 0
    for (next_ref_dir, solve_dir) in walk_order:
        if not walking_reference:
            next_ref_dir = ref_dir
        logging.info("Solving dir: {}".format(solve_dir))
        phase_di = phase_raw[:, next_ref_dir:next_ref_dir+1, ...]
        logging.info("Referencing dir: {}".format(next_ref_dir))
        phase_dd = phase_raw[:, solve_dir:solve_dir+1, ...] - phase_di

        # Npol, 1, Na, Nf, Nt
        Yimag = amp_smooth[:, solve_dir:solve_dir+1, ...] * np.sin(phase_dd)
        Yreal = amp_smooth[:, solve_dir:solve_dir+1, ...] * np.cos(phase_dd)
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
            dsk[str(c)] = (sequential_solve, Yreal[start:stop, :, :], Yimag[start:stop, :, :], freqs, os.path.join(working_dir,'proc_{:04d}'.format(proc_idx)))
            proc_idx += 1
            keys.append(str(c))
        logging.info("Running dask on {} processes".format(num_processes))
        results = get(dsk, keys, num_workers=num_processes)
        logging.info("Finished dask.")
        tec_mean = np.zeros((D, Nt))
        tec_uncert = np.zeros((D, Nt))
        obs_cov = np.zeros((D, 2*Nf, 2*Nf))

        for c, i in enumerate(range(0, D, D // num_processes)):
            start = i
            stop = min(i + (D // num_processes), D)
            tec_mean[start:stop, :] = results[c][0]
            tec_uncert[start:stop, :] = results[c][1]
            obs_cov[start:stop, :, :] = results[c][2]
        tec_mean = tec_mean.reshape((Npol, 1, Na, Nt))
        tec_uncert = tec_uncert.reshape((Npol, 1, Na, Nt))
        obs_cov = obs_cov.reshape((Npol, 1, Na, 2*Nf, 2*Nf))
        if walking_reference:
            logging.info("Re-referencing to {}".format(ref_dir))
            # phase_smooth[:, solve_dir:solve_dir+1, ...] = tec_mean[..., None, :] * tec_conv[:, None] + phase_di
            #Reference to ref_dir 0: tau_ij + tau_jk = tau_ik
            tec_mean_array[:, solve_dir:solve_dir+1,...] = tec_mean + tec_mean_array[:,next_ref_dir:next_ref_dir+1,...]
            tec_uncert_array[:, solve_dir:solve_dir+1, ...] = np.sqrt(tec_uncert**2 + tec_uncert_array[:, next_ref_dir:next_ref_dir+1, ...]**2)
            obs_cov_array[:, solve_dir:solve_dir+1, ...] = obs_cov
        else:
            tec_mean_array[:, solve_dir:solve_dir + 1, ...] = tec_mean
            tec_uncert_array[:, solve_dir:solve_dir + 1, ...] = tec_uncert
            obs_cov_array[:, solve_dir:solve_dir + 1, ...] = obs_cov

    # phase_smooth_uncert = np.abs(tec_conv[:, None] * tec_uncert_array[..., None, :])
    phase_model = tec_mean_array[..., None, :]*tec_conv[:, None] + phase_smooth[:, ref_dir:ref_dir+1, ...]
    res_real = amp_smooth * np.cos(phase_raw) - np.cos(phase_model)
    res_imag = amp_smooth * np.sin(phase_raw) - np.sin(phase_model)

    logging.info("Storing TEC and const")
    datapack.current_solset = 'directionally_referenced'
    # Npol, Nd, Na, Nf, Nt
    datapack.select(**select)
    datapack.tec = tec_mean_array
    datapack.weights_tec = tec_uncert_array
    logging.info("Storing obs. cov.")
    np.save(os.path.join(working_dir, "observational_covariance.npy"), obs_cov_array)
    logging.info("Done ddtec VI.")

    animate_datapack(merged_h5parm, os.path.join(working_dir, 'tec_plots'), num_processes=ncpu,
                     solset='directionally_referenced',
                     observable='tec', vmin=-60., vmax=60., plot_facet_idx=True,
                     labels_in_radec=True, plot_crosses=False, phase_wrap=False,
                     flag_outliers=False)

    animate_datapack(merged_h5parm, os.path.join(working_dir, 'tec_uncert_plots'), num_processes=ncpu,
                     solset='directionally_referenced',
                     observable='weights_tec', vmin=0., vmax=10., plot_facet_idx=True,
                     labels_in_radec=True, plot_crosses=False, phase_wrap=False,
                     flag_outliers=False)

    plot_results(Na, Nd, antenna_labels, working_dir, phase_model, phase_raw,
                 res_imag, res_real, tec_mean_array,obs_cov_array)

def wrap(p):
    return np.arctan2(np.sin(p), np.cos(p))

def plot_results(Na, Nd, antenna_labels, working_dir, phase_model,
                 phase_raw, res_imag, res_real, tec_mean_array, obs_cov_array):
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
            vmin = np.min(obs_cov_array[0,j,i,:,:])
            vmax = np.max(obs_cov_array[0,j,i,:,:])
            axs[1][1].imshow(obs_cov_array[0,j,i,:,:],origin='lower', vmin=vmin, vmax=vmax, aspect='auto', cmap='bone')
            axs[1][1].set_title("Obs. cov. est. [{:.2f}, {:.2f}]".format(vmin, vmax))
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
                        default=True, type="bool", required=False)


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
