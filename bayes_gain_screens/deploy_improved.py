from .model import AverageModel
from .datapack import DataPack
from typing import Union
from . import logging
from .misc import get_screen_directions_from_image, maybe_create_posterior_solsets, get_coordinates
from .outlier_detection import remove_outliers, Classifier
from . import TEC_CONV
from timeit import default_timer
import numpy as np
import tensorflow.compat.v1 as tf
import os, glob


class Deployment(object):
    def __init__(self, datapack: Union[DataPack, str],
                 ref_dir_idx=0,
                 ref_ant_idx=0,
                 tec_solset='directionally_referenced',
                 phase_solset='smoothed000',
                 flux_limit=0.01, max_N=250, min_spacing_arcmin=4.,
                 ref_image_fits: str = None,
                 ant=None, dir=None, time=None, freq=None, pol=slice(0, 1, 1),
                 block_size=1, working_dir='./deployment',
                 remake_posterior_solsets=False):

        working_dir = os.path.abspath(working_dir)
        os.makedirs(working_dir, exist_ok=True)
        os.chdir(working_dir)
        self.working_dir = working_dir

        logging.info("Using working directory {}".format(working_dir))
        logging.info("Flagging outliers in TEC")
        remove_outliers(False, False, True,
                        [datapack],
                        [ref_image_fits],
                        self.working_dir,
                        Classifier.flagging_models,
                        K=10,
                        L=7,
                        n_features=32,
                        batch_size=16
                        )

        if isinstance(datapack, DataPack):
            datapack = datapack.filename
        datapack = DataPack(datapack, readonly=False)

        logging.info("Getting reference phase data.")
        self.select = dict(ant=ant, dir=dir, time=time, freq=freq, pol=pol)
        datapack.current_solset = phase_solset
        datapack.select(**self.select)
        phase, _ = datapack.phase
        phase = phase.astype(np.float64)
        self.phase_di = phase[:, ref_dir_idx:ref_dir_idx + 1, ...]
        logging.info("Getting DDTEC data")
        datapack.current_solset = tec_solset
        tec, axes = datapack.tec
        tec = tec.astype(np.float64)
        tec_uncert, _ = datapack.weights_tec
        tec_uncert = tec_uncert.astype(np.float64)

        logging.info("Transposing data to (Nt, Na, Nd)")
        # Nd, Na, Nt -> Nt, Na, Nd
        tec = tec[0, ...].transpose((2, 1, 0))
        self.Nt, self.Na, self.Nd = tec.shape
        tec_uncert = tec_uncert[0, ...].transpose((2, 1, 0))

        logging.info("Setting clipping tec uncertainty to at least to 0.1 mTECU")
        tec_uncert = np.maximum(tec_uncert, 0.1)

        logging.info("Checking finiteness")
        if np.any(np.logical_not(np.isfinite(tec))):
            raise ValueError("Some tec are not finite.\n{}".format(
                np.where(np.logical_not(np.isfinite(tec)))))

        if np.any(np.isnan(tec_uncert)):
            raise ValueError("Some TEC uncerts are nan.\n{}".format(
                np.where(np.isnan(tec_uncert))))

        if np.any(np.logical_not(np.isfinite(phase))):
            raise ValueError("Some phases are not finite.\n{}".format(
                np.where(np.logical_not(np.isfinite(phase)))))

        logging.info("Creating posterior solsets.")
        _, seed_directions = datapack.get_directions(axes['dir'])
        seed_directions = np.stack([seed_directions.ra.rad, seed_directions.dec.rad], axis=1)
        screen_directions, _ = get_screen_directions_from_image(ref_image_fits, flux_limit=flux_limit, max_N=max_N,
                                                                min_spacing_arcmin=min_spacing_arcmin,
                                                                seed_directions=seed_directions,
                                                                fill_in_distance=8., fill_in_flux_limit=0.01)

        (self.screen_solset,) = maybe_create_posterior_solsets(datapack, phase_solset,
                                                               make_data_solset=False,
                                                               posterior_name='posterior',
                                                               screen_directions=screen_directions,
                                                               make_soltabs=['amplitude000', 'phase000', 'tec000'],
                                                               remake_posterior_solsets=remake_posterior_solsets)

        self.phase_solset = phase_solset
        self.tec_solset = tec_solset
        self.ref_dir_idx = ref_dir_idx
        self.ref_ant_idx = ref_ant_idx
        self.datapack = datapack
        self.tec = tec
        self.tec_uncert = tec_uncert

        self.block_size = block_size
        self.names = None
        self.models = None

    def run(self, model_generator, **model_kwargs):
        logging.info("Running the deployment.")
        t0 = default_timer()
        logging.info("Setting up coordinate transform ops.")
        # with tf.Session(graph=tf.Graph()) as tf_session:

        post_tec_mean_array = []
        post_tec_uncert_array = []
        weights_array = []
        log_marginal_likelihood_array = []

        init_hyperparams = None

        for t in range(0, self.Nt, self.block_size):

            start = t
            stop = min(self.Nt, t + self.block_size)
            block_size = stop - start
            mid_time = start + (stop - start) // 2
            logging.info("Beginning time block {} to {}".format(start, stop))
            logging.info("Getting the data and screen coordinates in the ENU frame.")

            feed_dict = {}
            self.datapack.current_solset = self.tec_solset
            self.datapack.select(time=slice(mid_time, mid_time + 1, 1))
            X, ref_location, ref_direction = get_coordinates(self.datapack, self.ref_ant_idx, self.ref_dir_idx)
            # Nd, Na, 6 -> Na, Nd, 6
            X = X[0, :, :, :].transpose((1, 0, 2))
            ref_location = ref_location[0, :]
            ref_direction = ref_direction[0, :]

            self.datapack.current_solset = self.screen_solset
            self.datapack.select(time=slice(mid_time, mid_time + 1, 1))
            X_screen, _, _ = get_coordinates(self.datapack, self.ref_ant_idx, self.ref_dir_idx)
            # Nd_, Na ->  Na, Nd_, 6
            X_screen = X_screen[0, :, :, :].transpose((1, 0, 2))

            with tf.Session(graph=tf.Graph()) as gp_session:
                # block_size, Na, Nd
                Y = self.tec[start:stop, :, :]
                # block_size, Na, Nd
                Y_var = np.square(self.tec_uncert[start:stop, :, :])

                logging.info("Building the hypothesis GP models.")
                self.models = model_generator(X, Y, Y_var, ref_location=ref_location, ref_direction=ref_direction,
                                              **model_kwargs)
                self.names = [m.caption for m in self.models]

                logging.info("Running with {} models:".format(len(self.models)))
                for i, m in enumerate(self.names):
                    logging.info("\t{} -> {}".format(i, m))

                model = AverageModel(self.models)
                logging.info("Optimising models")
                if init_hyperparams is not None:
                    logging.info("Transferring last hyperparams")
                    model.optimise(restart_points=init_hyperparams)
                else:
                    model.optimise()
                init_hyperparams = model.get_hyperparams()
                logging.info("Predicting posteriors and averaging")
                # batch_size, N / block_size, Na, Nd
                (weights, log_marginal_likelihoods), post_mean, post_var = model.predict_f(X_screen,
                                                                                           only_mean=False)
            # num_models, block_size, Na
            weights = np.reshape(weights, (-1, block_size, self.Na))
            log_marginal_likelihoods = np.reshape(log_marginal_likelihoods, (-1, block_size, self.Na))

            weights_array.append(weights)
            log_marginal_likelihood_array.append(log_marginal_likelihoods)
            post_tec_mean_array.append(post_mean)
            post_tec_uncert_array.append(np.sqrt(post_var))

        # num_models, Nt, Na
        weights_array = np.concatenate(weights_array, axis=1)
        log_marginal_likelihood_array = np.concatenate(log_marginal_likelihood_array, axis=1)

        # Nt, Na, Nd_screen -> Nd_screen, Na, Nt
        post_tec_mean_array = np.concatenate(post_tec_mean_array, axis=0).transpose((2, 1, 0))
        post_tec_uncert_array = np.concatenate(post_tec_uncert_array, axis=0).transpose((2, 1, 0))

        logging.info("Storing tec")
        self.datapack.current_solset = self.screen_solset
        self.datapack.select(**self.select)
        self.datapack.tec = post_tec_mean_array[None, ...]
        self.datapack.weights_tec = post_tec_uncert_array[None, ...]

        logging.info("Computing screen phase")
        axes = self.datapack.axes_phase
        _, freqs = self.datapack.get_freqs(axes['freq'])
        tec_conv = TEC_CONV / freqs
        # 1, Nd, Na, Nf, Nt
        post_phase_mean = post_tec_mean_array[None, ..., None, :] * tec_conv[:, None] + self.phase_di
        post_phase_std = np.abs(post_tec_uncert_array[None, ..., None, :] * tec_conv[:, None])

        logging.info("Computing screen amplitude with radial basis function")
        self.datapack.current_solset = self.screen_solset
        self.datapack.select(**self.select)
        axes = self.datapack.axes_phase
        _, screen_directions = self.datapack.get_directions(axes['dir'])

        self.datapack.current_solset = self.phase_solset
        self.datapack.select(**self.select)
        cal_amplitudes, _ = self.datapack.amplitude
        #Npol, Na, Nf, Nt, Nd
        cal_amplitudes = np.transpose(cal_amplitudes, (0, 2,3,4,1))
        Npol, Na, Nf, Nt, Nd = cal_amplitudes.shape
        #B, Nd
        cal_amplitudes = np.reshape(cal_amplitudes, (-1, cal_amplitudes.shape[-1]))
        axes = self.datapack.axes_phase
        _, cal_directions = self.datapack.get_directions(axes['dir'])
        cal_directions = np.stack([cal_directions.ra.deg, cal_directions.dec.deg], axis=1)
        screen_directions = np.stack([screen_directions.ra.deg, screen_directions.dec.deg], axis=1)

        #B, Nd
        cal_amplitudes = cal_amplitudes

        with tf.Session(graph=tf.Graph()) as smooth_sess:
            # N, 2
            x1_pl = tf.placeholder(tf.float32, shape=cal_directions.shape)
            # M, 2
            x2_pl = tf.placeholder(tf.float32, shape=screen_directions.shape)
            # B, Nd
            amp_pl = tf.placeholder(tf.float32, shape=cal_amplitudes.shape)
            smooth = tf.constant(0.1, tf.float32)
            # N, N
            A11 = tf.linalg.norm(tf.math.squared_difference(x1_pl[:, None, :], x1_pl[None, :, :]), axis=-1)
            A11 = A11 - tf.eye(tf.shape(A11)[-1], dtype=tf.float32) * smooth
            # M,N
            A21 = tf.linalg.norm(tf.math.squared_difference(x2_pl[:, None, :], x1_pl[None, :, :]), axis=-1)
            # b,N, 1
            w1 = tf.linalg.lstsq(tf.tile(A11[None, :, :], [cal_amplitudes.shape[0], 1, 1]), tf.math.log(amp_pl)[:, :, None],
                                 fast=False)
            # b, M
            screen_amps_pred = tf.math.exp(tf.linalg.matmul(A21, w1)[..., 0])
            screen_amps_pred = smooth_sess.run(screen_amps_pred, {x1_pl: cal_directions, x2_pl: screen_directions,
                                                                  amp_pl: cal_amplitudes})
        # Npol, Na, Nf, Nt, Nd_screen
        screen_amps_pred = screen_amps_pred.reshape((Npol, Na, Nf, Nt,-1))
        # Npol, Nd_screen, Na, Nf, Nt
        screen_amps_pred = screen_amps_pred.transpose((0,4,1,2,3))

        logging.info("Storing screen phases and amplitudes.")
        self.datapack.current_solset = self.screen_solset
        self.datapack.select(**self.select)
        self.datapack.phase = post_phase_mean
        self.datapack.weights_phase = post_phase_std
        self.datapack.amplitude = screen_amps_pred

        logging.info("Saving weights")
        np.savez(os.path.join(self.working_dir, 'weights.npz'), weights=weights_array, names=self.names,
                 log_marginal_likelihoods=log_marginal_likelihood_array)

        logging.info("Done. Time to run: {:.1f} seconds".format(default_timer() - t0))
