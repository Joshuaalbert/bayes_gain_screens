from .model import AverageModel
from .datapack import DataPack
from typing import List, Union
from .coord_transforms import ITRSToENUWithReferences_v2
from . import logging, angle_type, dist_type, float_type
from .misc import get_screen_directions_from_image, maybe_create_posterior_solsets, great_circle_sep
from .outlier_detection import filter_tec_dir
from scipy.ndimage import median_filter
from timeit import default_timer
import numpy as np
import tensorflow as tf
import os, glob

def smooth(v, axis=-1):
    out = np.zeros(v.shape)
    size = np.ones(len(v.shape), dtype=np.int)
    size[axis] = 3
    out[..., :-1] += np.cumsum(median_filter(np.diff(v[..., ::-1]), size), axis=axis)[..., ::-1]
    out += v[..., -1: ]
    out[..., 1:] += np.cumsum(median_filter(np.diff(v),size), axis=axis)
    out += v[..., 0:1]
    out /= 2.
    return out

class Deployment(object):
    def __init__(self, datapack: Union[DataPack, str], ref_dir_idx=0,
                 tec_solset='directionally_referenced', phase_solset='smoothed000',
                 flux_limit=0.05, max_N=250, min_spacing_arcmin=1.,
                 ref_image_fits: str = None, ant=None, dir=None, time=None, freq=None, pol=slice(0, 1, 1),
                 directional_deploy=True, block_size=1, debug=False, working_dir='./deployment', flag_directions=None,
                 flag_outliers=True,
                 constant_tec_uncert=None, constant_const_uncert=None, remake_posterior_solsets=False):

        self.debug = debug
        if self.debug:
            logging.info("In debug mode")

        cwd = os.path.abspath(working_dir)
        os.makedirs(cwd, exist_ok=True)
        os.chdir(working_dir)
        # run_idx = len(glob.glob(os.path.join(cwd, 'run_*')))
        # cwd = os.path.join(cwd, 'run_{:03d}'.format(run_idx))
        # os.makedirs(cwd, exist_ok=True)
        self.cwd = cwd

        if debug:
            self.debug_dir = os.path.join(self.cwd, 'debug')
            os.makedirs(self.debug_dir, exist_ok=True)
        logging.info("Using working directory {}".format(cwd))

        if isinstance(datapack, DataPack):
            datapack = datapack.filename
        datapack = DataPack(datapack, readonly=False)

        logging.info("Getting TEC, const, and reference phase data from datapack.")
        self.select = dict(ant=ant, dir=dir, time=time, freq=freq, pol=pol)
        datapack.current_solset = phase_solset
        datapack.select(**self.select)
        phase, _ = datapack.phase
        phase = phase.astype(np.float64)
        self.phase_di = phase[:, ref_dir_idx:ref_dir_idx + 1, ...]
        datapack.current_solset = tec_solset
        tec, axes = datapack.tec
        tec = tec.astype(np.float64)
        tec_uncert, _ = datapack.weights_tec
        tec_uncert = tec_uncert.astype(np.float64)
        const, axes = datapack.const
        const = const.astype(np.float64)
        const_uncert, _ = datapack.weights_const
        const_uncert = const_uncert.astype(np.float64)

        const = median_filter(const, size=(1, 1, 1, 31))
        
        _, data_directions = datapack.get_directions(axes['dir'])
        data_directions = np.stack([data_directions.ra.rad, data_directions.dec.rad],
                                   axis=1)
        if flag_outliers:
            logging.info("Flagging outliers in TEC")
            tec_uncert, _ = filter_tec_dir(tec[0,...], data_directions, init_y_uncert=tec_uncert[0,...], min_res=8., function='multiquadric')
            tec_uncert = tec_uncert[None,...]
            logging.info("Fraction flagged: {:.3f}".format(np.sum(tec_uncert==np.inf)/tec_uncert.size))
        if flag_directions is not None:
            tec_uncert[:, flag_directions, ...] = np.inf
            const_uncert[:, flag_directions, ...] = np.inf

        logging.info("Number flagged: {} from {}".format(np.sum(np.isinf(tec_uncert)), tec_uncert.size))
        logging.info("Transposing data")
        if directional_deploy:
            # Nd, Na, Nt -> Nt, Na, Nd
            tec = tec[0, ...].transpose((2, 1, 0))
            self.Nt, self.Na, self.Nd = tec.shape
            tec_uncert = tec_uncert[0, ...].transpose((2, 1, 0))
            if self.debug:
                logging.info("tec shape: {} should be (Nt, Na, Nd)".format(tec.shape))
            # Nd, Na, Nt -> Nt, Na, Nd
            const = const[0, ...].transpose((2, 1, 0))
            self.Nt, self.Na, self.Nd = const.shape
            const_uncert = const_uncert[0, ...].transpose((2, 1, 0))
            if self.debug:
                logging.info("const shape: {} should be (Nt, Na, Nd)".format(const.shape))
        else:
            # Nt, Nd, Na
            tec = tec[0, ...].transpose((2, 0, 1))
            self.Nt, self.Nd, self.Na = tec.shape
            tec_uncert = tec_uncert[0, ...].transpose((2, 0, 1))
            if self.debug:
                logging.info("tec shape: {} should be (Nt, Nd, Na)".format(tec.shape))
            # Nt, Nd, Na
            const = const[0, ...].transpose((2, 0, 1))
            self.Nt, self.Nd, self.Na = const.shape
            const_uncert = const_uncert[0, ...].transpose((2, 0, 1))
            if self.debug:
                logging.info("const shape: {} should be (Nt, Nd, Na)".format(const.shape))
        if constant_tec_uncert is not None:
            logging.info("Setting all non-flagged TEC uncert to {}".format(constant_tec_uncert))
            tec_uncert = np.where(tec_uncert == np.inf, np.inf, constant_tec_uncert)
        logging.info("Setting minimum tec uncertainty to 0.5 mTECU")
        tec_uncert = np.maximum(tec_uncert, 0.5, tec_uncert)
        if constant_const_uncert is not None:
            logging.info("Setting all non-flagged const uncert to {}".format(constant_const_uncert))
            const_uncert = np.where(const_uncert == np.inf, np.inf, constant_const_uncert)
        logging.info("Setting minimum const uncertainty to 0.5 mTECU")
        const_uncert = np.maximum(const_uncert, 0.01, const_uncert)
        logging.info("Checking finiteness")
        if np.any(np.logical_not(np.isfinite(tec))):
            raise ValueError("Some tec are not finite.\n{}".format(
                np.where(np.logical_not(np.isfinite(tec)))))
        if np.any(np.logical_not(np.isfinite(const))):
            raise ValueError("Some const are not finite.\n{}".format(
                np.where(np.logical_not(np.isfinite(const)))))
        if np.any(np.isnan(tec_uncert)):
            raise ValueError("Some TEC uncerts are nan.\n{}".format(
                np.where(np.isnan(tec_uncert))))
        if np.any(np.isnan(const_uncert)):
            raise ValueError("Some const uncerts are nan.\n{}".format(
                np.where(np.isnan(const_uncert))))
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
                                                               make_soltabs=['phase000', 'tec000', 'const000',
                                                                             'amplitude000'],
                                                               remake_posterior_solsets=remake_posterior_solsets)

        logging.info("Getting time, dir, and ant coordinates.")
        _, times = datapack.get_times(axes['time'])
        Xt = (times.mjd * 86400.)[:, None]
        _, directions = datapack.get_directions(axes['dir'])
        Xd = np.stack([directions.ra.to(angle_type).value, directions.dec.to(angle_type).value], axis=1)
        self.ref_dir = Xd[ref_dir_idx, :]
        _, antennas = datapack.get_antennas(axes['ant'])
        Xa = antennas.cartesian.xyz.to(dist_type).value.T
        logging.info("Getting screen coordinates.")
        datapack.current_solset = self.screen_solset
        datapack.select(**self.select)
        axes = datapack.axes_phase
        _, screen_directions = datapack.get_directions(axes['dir'])
        Xd_screen = np.stack([screen_directions.ra.to(angle_type).value, screen_directions.dec.to(angle_type).value],
                             axis=1)

        if self.debug:
            with np.printoptions(precision=2):
                logging.info("directions:\n{}".format(Xd))
                logging.info("screen directions:\n{}".format(Xd_screen))
                logging.info("antennas:\n{}".format(Xa))

        self.phase_solset = phase_solset
        self.tec_solset = tec_solset
        self.directional_deploy = directional_deploy
        self.ref_ant = Xa[0, :]
        self.Nd_screen = Xd_screen.shape[0]
        self.datapack = datapack
        self.Xa = Xa
        self.Xd = Xd
        self.Xd_screen = Xd_screen
        self.Xt = Xt
        self.tec = tec
        self.tec_uncert = tec_uncert
        self.const = const
        self.const_uncert = const_uncert
        self.block_size = block_size
        self.names = None
        self.models = None

    def run(self, model_generator, **model_kwargs):
        logging.info("Running the deployment.")
        t0 = default_timer()
        logging.info("Setting up coordinate transform ops.")
        with tf.Session(graph=tf.Graph()) as tf_session:
            logging.info("Creating coordinate transform")
            coord_transform = ITRSToENUWithReferences_v2(self.ref_ant, self.ref_dir, self.ref_ant)
            Xt_pl = tf.placeholder(float_type, shape=[1, 1])
            Xd_pl = tf.placeholder(float_type, shape=self.Xd.shape)
            Xd_screen_pl = tf.placeholder(float_type, shape=self.Xd_screen.shape, )
            Xa_pl = tf.placeholder(float_type, shape=self.Xa.shape)
            tf_X, tf_ref_ant, tf_ref_dir = coord_transform(Xt_pl, Xd_pl, Xa_pl)
            tf_X_screen, _, _ = coord_transform(Xt_pl, Xd_screen_pl, Xa_pl)

            if self.directional_deploy:
                # Nd, 3
                tf_X = tf_X[0, :, 0, :3]
                # Nd_screen, 3
                tf_X_screen = tf_X_screen[0, :, 0, :3]
            else:
                # 1 * Nd * Na, 6
                tf_X = tf.reshape(tf_X, (-1, 6))
                # Nt * Nd_screen * Na, 6
                tf_X_screen = tf.reshape(tf_X_screen, (-1, 6))

            post_mean_array = []
            post_std_array = []
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
                # T*Na, N
                feed_dict[Xa_pl] = self.Xa
                feed_dict[Xd_pl] = self.Xd
                feed_dict[Xd_screen_pl] = self.Xd_screen
                feed_dict[Xt_pl] = self.Xt[mid_time:mid_time + 1, :]

                X, X_screen, ref_direction, ref_location = tf_session.run([tf_X, tf_X_screen, tf_ref_dir, tf_ref_ant],
                                                                          feed_dict)
                if self.debug:
                    with np.printoptions(precision=2):
                        logging.info("ref direction: {}".format(ref_direction))
                        logging.info("ref location: {}".format(ref_location))
                        logging.info("X:\n{}".format(X))
                        logging.info("X_screen:\n{}".format(X_screen))

                with tf.Session(graph=tf.Graph()) as gp_session:
                    if self.directional_deploy:
                        # block_size, Na, Nd
                        Y = self.tec[start:stop, :, :]
                        # block_size, Na, Nd
                        Y_var = np.square(self.tec_uncert[start:stop, :, :])

                        if self.debug:
                            self.debug_plot_data(t, self.tec[start:stop, :, :], self.tec_uncert[start:stop, :, :])
                            with np.printoptions(precision=2):
                                logging.info("Y:\n{}".format(Y))
                                logging.info("Y_var:\n{}".format(Y_var))
                        logging.info("Data shape should be (block_size, Na, Nd)")
                        logging.info("Shapes:\nX: {}, Y: {}, Y_var: {}, X_screen: {} ref_dir: {}".format(X.shape, Y.shape,
                                                                                                Y_var.shape,
                                                                                                X_screen.shape,
                                                                                                ref_direction))
                        logging.info("Building the GP models.")
                        self.models = model_generator(X, Y, Y_var, ref_direction, reg_param=1., parallel_iterations=10,
                                                      **model_kwargs)
                        self.names = [m.name for m in self.models]

                    else:
                        # block_size, Nd, Na
                        Y = self.tec[start:stop, :, :]
                        # block_size, Nd*Na
                        Y = Y.reshape((-1, self.Nd * self.Na))
                        # block_size, Nd*Na
                        Y_var = np.square(self.tec_uncert[start:stop, :, :]).reshape((-1, self.Nd * self.Na))
                        if self.debug:
                            with np.printoptions(precision=2):
                                logging.info("Y:\n{}".format(Y))
                                logging.info("Y_var:\n{}".format(Y_var))
                        logging.info("Data shape should be (block_size, Nd*Na)")
                        logging.info(
                            "Shapes:\nX: {}, Y: {}, Y_var: {}, X_screen: {} ref_dir: {} ref_ant: {}".format(X.shape,
                                                                                                            Y.shape,
                                                                                                            Y_var.shape,
                                                                                                            X_screen.shape,
                                                                                                            ref_direction,
                                                                                                            ref_location))
                        self.models = model_generator(X, Y, Y_var, ref_direction, ref_location, reg_param=1.,
                                                      parallel_iterations=10, **model_kwargs)
                        self.names = [m.name for m in self.models]
                    logging.info("Running with {} models:".format(len(self.models)))
                    for i, m in enumerate(self.models):
                        logging.info("\t{} -> {}".format(i, m.name))
                    if self.debug:
                        logging.info("Models:\n{}".format(self.names))
                    model = AverageModel(self.models, debug=self.debug)
                    if init_hyperparams is not None:
                        logging.info("Transferring last hyperparams")
                        model.set_hyperparams(init_hyperparams)
                    logging.info("Optimising models")
                    model.optimise()
                    init_hyperparams = model.get_hyperparams()
                    logging.info("Predicting posteriors and averaging")
                    # batch_size, N
                    (weights, log_marginal_likelihoods), post_mean, post_var = model.predict_f(X_screen,
                                                                                               only_mean=False)
                if self.directional_deploy:
                    # num_models, block_size, Na
                    weights = np.reshape(weights, (-1, block_size, self.Na))
                    log_marginal_likelihoods = np.reshape(log_marginal_likelihoods, (-1, block_size, self.Na))
                    # block_size, Na, Nd -> block_size, Nd_screen, Na
                    post_mean = post_mean.reshape((block_size, self.Na, self.Nd_screen)).transpose((0, 2, 1))
                    post_var = post_var.reshape((block_size, self.Na, self.Nd_screen)).transpose((0, 2, 1))
                    if self.debug:
                        self.debug_plot_posterior(t, post_mean, post_var)
                else:
                    # num_models, block_size
                    weights = np.reshape(weights, (-1, block_size))
                    log_marginal_likelihoods = np.reshape(log_marginal_likelihoods, (-1, block_size))
                    # block_size, Nd, Na -> block_size, Nd_screen, Na
                    post_mean = post_mean.reshape((block_size, self.Nd_screen, self.Na))
                    post_var = post_var.reshape((block_size, self.Nd_screen, self.Na))
                    if self.debug:
                        self.debug_plot_posterior(t, post_mean, post_var)


                weights_array.append(weights)
                log_marginal_likelihood_array.append(log_marginal_likelihoods)
                post_mean_array.append(post_mean)
                post_std_array.append(np.sqrt(post_var))

            if self.directional_deploy:
                # num_models, Nt, Na
                weights_array = np.concatenate(weights_array, axis=1)
                log_marginal_likelihood_array = np.concatenate(log_marginal_likelihood_array, axis=1)
            else:
                # num_models, Nt
                weights_array = np.concatenate(weights_array, axis=1)
                log_marginal_likelihood_array = np.concatenate(log_marginal_likelihood_array, axis=1)

            if self.directional_deploy:
                #Nt, Na, Nd -> Nd, Na, Nt
                const_mean = self.const.transpose((2,1,0))
            else:
                #Nt, Nd, Na -> Nd, Na, Nt
                const_mean = self.const.transpose((1,2,0))

            # Nt, Nd, Na -> Nd_screen, Na, Nt
            post_mean_array = np.concatenate(post_mean_array, axis=0).transpose((1, 2, 0))
            post_std_array = np.concatenate(post_std_array, axis=0).transpose((1, 2, 0))
            logging.info("Storing tec")
            self.datapack.current_solset = self.screen_solset
            self.datapack.select(**self.select)
            self.datapack.tec = post_mean_array[None, ...]
            self.datapack.weights_tec = post_std_array[None, ...]
            logging.info("Getting NN indices")
            dir_idx = [np.argmin(
                great_circle_sep(self.Xd[:, 0], self.Xd[:, 1], ra, dec) for (ra, dec) in zip(self.Xd_screen[:, 0], self.Xd_screen[:, 1]))]
            logging.info("Getting NN const")
            #Nd_screen, Na, Nt
            const_NN = const_mean[dir_idx, :, :]
            logging.info("Storing const")
            self.datapack.const = const_NN[None,...]

            logging.info("Computing screen phase")
            axes = self.datapack.axes_phase
            _, freqs = self.datapack.get_freqs(axes['freq'])
            tec_conv = -8.4479745e6 / freqs
            #1, Nd, Na, Nf, Nt
            post_phase_mean = post_mean_array[None, ..., None, :] * tec_conv[:, None] + self.phase_di# + const_NN[None,..., None, :]
            post_phase_std = np.abs(post_std_array[None, ..., None, :] * tec_conv[:, None])

            logging.info("Replacing calbrator phases with smoothed phases")
            self.datapack.current_solset = self.phase_solset
            self.datapack.select(**self.select)
            smoothed_phase, _ = self.datapack.phase
            smoothed_phase_uncert, _ = self.datapack.weights_phase
            post_phase_mean[:,:self.Nd,...] = smoothed_phase
            post_phase_std[:,:self.Nd,...] = smoothed_phase_uncert

            logging.info("Storing screen phases.")
            self.datapack.current_solset = self.screen_solset
            self.datapack.select(**self.select)
            self.datapack.phase = post_phase_mean
            self.datapack.weights_phase = post_phase_std
            logging.info("NN interp of amplitudes.")
            self.datapack.current_solset = self.phase_solset
            self.datapack.select(**self.select)
            amplitude, _ = self.datapack.amplitude
            logging.info("Storing amplitudes.")
            self.datapack.current_solset = self.screen_solset
            self.datapack.select(**self.select)
            self.datapack.amplitude = amplitude[:, dir_idx, ...]
            logging.info("Saving weights")
            np.savez(os.path.join(self.cwd, 'weights.npz'), weights = weights_array, names = self.names,
                     log_marginal_likelihoods = log_marginal_likelihood_array)
            if self.debug:
                with np.printoptions(formatter={'float': '{: 0.3f}'.format}):
                    for idx, weights in enumerate(list(weights_array)):
                        logging.info("Model {}\n{}".format(self.names[idx], weights))

        logging.info("Done. Time to run: {:.1f} seconds".format(default_timer() - t0))

    def debug_plot_posterior(self, t, post_mean, post_var):
        num_plots = int(np.sqrt(self.Na))
        import pylab as plt
        from bayes_gain_screens.plotting import plot_vornoi_map
        fig, axs = plt.subplots(num_plots, num_plots, sharex=True, sharey=True, figsize=(num_plots * 3, num_plots * 3))
        c = 0
        for i in range(num_plots):
            for j in range(num_plots):
                if c >= self.Na:
                    continue

                plot_vornoi_map(self.Xd_screen,
                                post_mean[0, :, c],
                                axs[i][j],
                                radius=2 * np.pi,
                                cmap=plt.cm.coolwarm)
                c += 1
        plt.savefig(os.path.join(self.debug_dir, "posterior_mean_{:03d}.png".format(t)))
        plt.close('all')

        fig, axs = plt.subplots(num_plots, num_plots, sharex=True, sharey=True, figsize=(num_plots * 3, num_plots * 3))
        c = 0
        for i in range(num_plots):
            for j in range(num_plots):
                if c >= self.Na:
                    continue

                plot_vornoi_map(self.Xd_screen,
                                np.sqrt(post_var[0, :, c]),
                                axs[i][j],
                                radius=2 * np.pi,
                                cmap=plt.cm.coolwarm)
                c += 1
        plt.savefig(os.path.join(self.debug_dir, "data_std_{:03d}.png".format(t)))
        plt.close('all')
    
    def debug_plot_data(self,t, data_mean, data_std):
        np.savez(os.path.join(self.cwd, 'data_{:03d}.npz'.format(t)), mean=data_mean, std=data_std, Xd=self.Xd)
        return
        data_std = np.copy(data_std)
        data_std[np.isinf(data_std)] = np.nan
        num_plots = int(np.sqrt(self.Na))
        import pylab as plt
        from bayes_gain_screens.plotting import plot_vornoi_map
        fig, axs = plt.subplots(num_plots, num_plots, sharex=True, sharey=True, figsize=(num_plots * 3, num_plots * 3))
        c = 0
        for i in range(num_plots):
            for j in range(num_plots):
                if c >= self.Na:
                    break

                plot_vornoi_map(self.Xd,
                                data_mean[0, c, :],
                                axs[i][j],
                                radius=None,
                                relim=True,
                                cmap=plt.cm.coolwarm)
                c += 1
        plt.savefig(os.path.join(self.debug_dir, "data_mean_{:03d}.png".format(t)))
        plt.close('all')

        fig, axs = plt.subplots(num_plots, num_plots, sharex=True, sharey=True, figsize=(num_plots * 3, num_plots * 3))
        c = 0
        for i in range(num_plots):
            for j in range(num_plots):
                if c >= self.Na:
                    break

                plot_vornoi_map(self.Xd,
                                data_std[0, c, :],
                                axs[i][j],
                                radius=2 * np.pi,
                                cmap=plt.cm.coolwarm)
                c += 1
        plt.savefig(os.path.join(self.debug_dir, "dataerior_std_{:03d}.png".format(t)))
        plt.close('all')
