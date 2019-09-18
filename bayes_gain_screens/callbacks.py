from .datapack import DataPack
import tensorflow as tf
import numpy as np
import os
from . import logging
from .plotting import DatapackPlotter
import pylab as plt



def callback_sequence(callbacks, args, async_=False):

    if async_:
        ops = []
        for arg, callback in zip(args, callbacks):
            if not isinstance(arg, (tuple, list)):
                arg = [arg]
            ops.append(callback(*arg))
        return tf.group(ops)

    lock = [tf.no_op()]
    store_ops = []
    for arg, callback in zip(args, callbacks):
        if not isinstance(arg, (tuple, list)):
            arg = [arg]
        with tf.control_dependencies(lock):
            store_ops.append(callback(*arg))
            lock = store_ops[-1]
    with tf.control_dependencies(lock):
        return tf.no_op()


class Callback(object):
    def __init__(self, *args, controls=None, **kwargs):
        self._output_dtypes = None
        self._name = 'Callback'
        self._squeeze = False
        self.controls = controls
        self.callback_func = self.generate(*args, **kwargs)

    @property
    def controls(self):
        return self._controls

    @controls.setter
    def controls(self, value):
        if value is None:
            self._controls = None
            return
        if not isinstance(value, (list, tuple)):
            value = [value]
        self._controls = list(value)

    @property
    def squeeze(self):
        return self._squeeze

    @squeeze.setter
    def squeeze(self, value):
        self._squeeze = value

    def generate(self, *args, **kwargs):
        raise NotImplementedError("Must subclass")

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = str(value)

    @property
    def output_dtypes(self):
        if self._output_dtypes is None:
            raise ValueError("Output dtype should be a list of output dtypes.")
        return self._output_dtypes

    @output_dtypes.setter
    def output_dtypes(self, value):
        if not isinstance(value, (list, tuple)):
            raise ValueError("output dtypes must be a list or tuple")
        self._output_dtypes = value

    def __call__(self, *Tin):
        squeeze = len(self.output_dtypes) == 1
        def py_func(*Tin):
            result = self.callback_func(*[t.numpy() for t in Tin])
            if not isinstance(result, (list,tuple)):
                result = [result]

            if len(result) != len(self.output_dtypes):
                raise ValueError("Len of py_function result {} not equal to number of output dtypes {}".format(len(result), len(self.output_dtypes)))
            if squeeze and self.squeeze:
                return result[0]
            return result
        if self.controls is not None:
            with tf.control_dependencies(self.controls):
                if squeeze and self.squeeze:
                    return tf.py_function(py_func, Tin, self.output_dtypes[0], name=self.name)
                return tf.py_function(py_func, Tin, self.output_dtypes, name=self.name)
        else:
            if squeeze and self.squeeze:
                return tf.py_function(py_func, Tin, self.output_dtypes[0], name=self.name)
            return tf.py_function(py_func, Tin, self.output_dtypes, name=self.name)

class Chain(Callback):
    def __init__(self, *callbacks, async_ = False):
        for cb in callbacks:
            if not isinstance(cb, Callback):
                raise ValueError("All inputs should be Callbacks, got {}".format(type(cb)))
        self._callbacks = callbacks
        self._async = async_

    def __call__(self, *args):
        if self._async:
            ops = []
            for arg, callback in zip(args, self._callbacks):
                if not isinstance(arg, (tuple, list)):
                    arg = [arg]
                ops.append(callback(*arg))
            return tf.group(ops)

        lock = [tf.no_op()]
        store_ops = []
        for arg, callback in zip(args, self._callbacks):
            if not isinstance(arg, (tuple, list)):
                arg = [arg]
            with tf.control_dependencies(lock):
                store_ops.append(callback(*arg))
                lock = store_ops[-1]
        with tf.control_dependencies(lock):
            return tf.no_op()


class SummarySendCallback(Callback):
    """
    Callback to submit summaries in-graph mode.
    """
    def __init__(self, logdir):
        super(SummarySendCallback, self).__init__(logdir=logdir)

    def generate(self, logdir):

        self.output_dtypes = [tf.int64]
        self.name = 'SummarySendCallback'
        filewriter = tf.summary.FileWriter(logdir,
                                           graph=tf.get_default_graph(),
                                           flush_secs=30)


        def store(i, *summaries):
            for summary in summaries:
                filewriter.add_summary(summary, i)
            return [np.array(len(summaries),dtype=np.int64)]
        return store

class DatapackStoreCallback(Callback):
    def __init__(self, datapack, solset, soltab, perm=(0,2,3,1),lock=None,index_map=None,**selection):
        super(DatapackStoreCallback, self).__init__(datapack=datapack,
                                                    solset=solset,
                                                    soltab=soltab,
                                                    perm=perm,
                                                    lock=lock,
                                                    index_map=index_map,
                                                    **selection)

    def generate(self, datapack, solset, soltab, perm, lock, index_map, **selection):

        if not isinstance(datapack, str):
            datapack = datapack.filename

        selection.pop('time',None)

        self.output_dtypes = [tf.int64]
        self.name = 'DatapackStoreCallback'

        def store(time_start, time_stop, array):
            time_start = index_map[time_start]
            time_stop = index_map[time_stop - 1] + 1
            if lock is not None:
                lock.acquire()
            with DataPack(datapack,readonly=False) as dp:
                dp.current_solset = solset
                dp.select(time=slice(time_start, time_stop, 1), **selection)
                dp.__setattr__(soltab, np.transpose(array, perm))#, dir=dir_sel, ant=ant_sel, freq=freq_sel, pol=pol_sel
            if lock is not None:
                lock.release()
            return [np.array(array.__sizeof__(),dtype=np.int64)]

        return store

class GetLearnIndices(Callback):
    def __init__(self, dist_cutoff=0.3):
        super(GetLearnIndices, self).__init__(dist_cutoff=dist_cutoff)

    def generate(self, dist_cutoff):
        self.output_dtypes = [tf.int64]
        self.name = 'GetLearnIndices'
        def get_learn_indices(X):
            """Get the indices of non-redundant antennas
            :param X: np.array, float64, [N, 3]
                Antenna locations
            :param cutoff: float
                Mark redundant if antennas within this in km
            :return: np.array, int64
                indices such that all antennas are at least cutoff apart
            """
            N = X.shape[0]
            Xa, inverse = np.unique(X, return_inverse=True, axis=0)
            Na = len(Xa)
            keep = []
            for i in range(Na):
                if np.all(np.linalg.norm(Xa[i:i + 1, :] - Xa[keep, :], axis=1) > dist_cutoff):
                    keep.append(i)
            logging.info("Training on antennas: {}".format(keep))
            return [(np.where(np.isin(inverse, keep, assume_unique=True))[0]).astype(np.int64)]
        return get_learn_indices


class StoreHyperparameters(Callback):
    def __init__(self, store_file):
        super(StoreHyperparameters, self).__init__(store_file=store_file)

    def generate(self, store_file):

        if not isinstance(store_file, str):
            raise ValueError("store_file should be str {}".format(type(store_file)))

        store_file=os.path.abspath(store_file)

        np.savez(store_file, times=np.array([]), amp=np.array([]), y_sigma=np.array([]), variance=np.array([]), lengthscales=np.array([]), a=np.array([]), b=np.array([]), timescale=np.array([]),
                 )


        self.output_dtypes = [tf.int64]
        self.name = 'StoreHyperparameters'

        def store(time, hyperparams, y_sigma, amp):
            data = np.load(store_file)
            #must match the order in the Target
            variance, lengthscales, a, b, timescale = np.reshape(hyperparams, (-1,))

            times = np.array([time] + list(data['times']))
            y_sigma = np.array([np.reshape(y_sigma,(-1,))] + list(data['y_sigma']))
            amp = np.array([np.reshape(amp, (-1,))] + list(data['amp']))
            variance = np.array([variance] + list(data['variance']))
            lengthscales = np.array([lengthscales] + list(data['lengthscales']))
            a = np.array([a] + list(data['a']))
            b = np.array([b] + list(data['b']))

            timescale = np.array([timescale] + list(data['timescale']))

            np.savez(store_file,
                     times=times,
                     y_sigma=y_sigma,
                     amp=amp,
                     variance=variance,
                     lengthscales=lengthscales,
                     a=a,
                     b=b,
                     timescale=timescale
                     )

            return [np.array(len(times),dtype=np.int64)]

        return store

class StoreHyperparametersV2(Callback):
    def __init__(self, store_file):
        super(StoreHyperparametersV2, self).__init__(store_file=store_file)

    def generate(self, store_file):

        if not isinstance(store_file, str):
            raise ValueError("store_file should be str {}".format(type(store_file)))

        store_file=os.path.abspath(store_file)
        if not os.path.exists(store_file):
            np.savez(store_file, times=np.array([]), amp=np.array([]), y_sigma=np.array([]), variance=np.array([]), lengthscales=np.array([]), a=np.array([]), b=np.array([]), timescale=np.array([]),
                     clock_scale=np.array([]))


        self.output_dtypes = [tf.int64]
        self.name = 'StoreHyperparametersV2'

        def store(time, amp, lengthscales, a, b, timescale, clock_scale, y_sigma):
            data = np.load(store_file)

            times = np.array([time] + list(data['times']))
            y_sigma = np.array([np.reshape(y_sigma,(-1,))] + list(data['y_sigma']))
            amp = np.array([np.reshape(amp, (-1,))] + list(data['amp']))
            lengthscales = np.array([lengthscales.reshape((-1,))] + list(data['lengthscales']))
            a = np.array([a.reshape((-1,))] + list(data['a']))
            b = np.array([b.reshape((-1,))] + list(data['b']))
            timescale = np.array([timescale.reshape((-1,))] + list(data['timescale']))
            clock_scale = np.array([clock_scale.reshape((-1,))] + list(data['clock_scale']))

            np.savez(store_file,
                     times=times,
                     y_sigma=y_sigma,
                     amp=amp,
                     lengthscales=lengthscales,
                     a=a,
                     b=b,
                     timescale=timescale,
                     clock_scale=clock_scale
                     )

            return [np.array(len(times),dtype=np.int64)]

        return store

class StoreHyperparametersV3(Callback):
    def __init__(self, store_file):
        super(StoreHyperparametersV3, self).__init__(store_file=store_file)

    def generate(self, store_file):

        if not isinstance(store_file, str):
            raise ValueError("store_file should be str {}".format(type(store_file)))

        store_file=os.path.abspath(store_file)
        if not os.path.exists(store_file):
            np.savez(store_file, times=np.array([]), amp=np.array([]), lengthscales=np.array([]), a=np.array([]), b=np.array([]))

        self.output_dtypes = [tf.int64]
        self.name = 'StoreHyperparametersV3'

        def store(time, amp, lengthscales, a, b):
            data = np.load(store_file)

            times = np.array([time] + list(data['times']))
            amp = np.array([np.reshape(amp, (-1,))] + list(data['amp']))
            lengthscales = np.array([lengthscales.reshape((-1,))] + list(data['lengthscales']))
            a = np.array([a.reshape((-1,))] + list(data['a']))
            b = np.array([b.reshape((-1,))] + list(data['b']))

            np.savez(store_file,
                     times=times,
                     amp=amp,
                     lengthscales=lengthscales,
                     a=a,
                     b=b
                     )

            return [np.array(len(times),dtype=np.int64)]

        return store

class PlotResults(Callback):
    def __init__(self, hyperparam_store, datapack, solset, lock=None, posterior_name='posterior', plot_directory='./plots', **selection):
        super(PlotResults, self).__init__(hyperparam_store=hyperparam_store,
                                          lock=lock,
                                          datapack=datapack,
                                          solset=solset,
                                          posterior_name=posterior_name,
                                          plot_directory=plot_directory,
                                          **selection)

    def generate(self, hyperparam_store, datapack, solset, lock, posterior_name, plot_directory, **selection):
        self.output_dtypes = [tf.int64]
        self.name = 'PlotResults'

        if not isinstance(plot_directory, str):
            raise ValueError("plot_directory should be str {}".format(type(plot_directory)))
        plot_directory = os.path.abspath(plot_directory)
        fig_directory = os.path.join(plot_directory,'phase_screens')
        # bayes_directory = os.path.join(plot_directory, 'bayes_hyperparmeters')
        os.makedirs(fig_directory,exist_ok=True)
        # os.makedirs(bayes_directory, exist_ok=True)
        dp = DatapackPlotter(datapack)

        def plot_results(index_start, index_end):
            """Get the indices of non-redundant antennas
            :param X: np.array, float64, [N, 3]
                Antenna locations
            :param cutoff: float
                Mark redundant if antennas within this in km
            :return: np.array, int64
                indices such that all antennas are at least cutoff apart
            """

            data = np.load(hyperparam_store)
            keys = ['amp','y_sigma','variance', 'lengthscales', 'a', 'b', 'timescale', 'clock_scale']
            if lock is not None:
                lock.acquire()
            fig, axs = plt.subplots(len(keys),1,sharex='all', figsize=(6,len(keys)*2))
            for i,key in enumerate(keys):
                ax = axs[i]
                if key in ['amp','y_sigma']:
                    # for t,d in zip(data['times'],data['y_sigma']):
                    ax.boxplot(data[key].T,positions=data['times'])
                    ax.set_title(key)
                else:
                    ax.scatter(data['times'], data[key], label=key)
                ax.legend()
            plt.savefig(os.path.join(plot_directory,'hyperparameters.png'))
            plt.close('all')
            if lock is not None:
                lock.release()

            # keys = ['amp', 'y_sigma']
            # if lock is not None:
            #     lock.acquire()
            # fig, axs = plt.subplots(len(keys), 1, figsize=(6, len(keys) * 2))
            # for i, key in enumerate(keys):
            #     ax = axs[i]
            #     ax.hist(data[key][-1], bins=max(10, int(np.sqrt(np.size(data[key][-1])))), label=key)
            #     ax.legend()
            # plt.savefig(os.path.join(bayes_directory, 'bayesian_hyperparameters_{:04d}_{:04d}.png'.format(index_start, index_end)))
            # plt.close('all')
            # if lock is not None:
            #     lock.release()

            fignames = [os.path.join(fig_directory, "fig-{:04d}_{}_phase.png".format(i,solset)) for i in
                        range(index_start, index_end, 1)]
            if lock is not None:
                lock.acquire()
            dp.plot(ant_sel=selection.get('ant',None),
                    time_sel=slice(index_start, index_end, 1),
                    freq_sel=selection.get('freq',None),
                    dir_sel=selection.get('dir',None),
                    pol_sel=selection.get('pol', slice(0,1,1)),
                    fignames=fignames,
                    observable='phase',
                    phase_wrap=True,
                    plot_facet_idx=True,
                    labels_in_radec=True,
                    solset=solset)
            plt.close('all')
            if lock is not None:
                lock.release()

            data_posterior = "data_{}".format(posterior_name)
            fignames = [os.path.join(fig_directory, "fig-{:04d}_{}_phase.png".format(i, data_posterior)) for i in
                        range(index_start, index_end, 1)]
            if lock is not None:
                lock.acquire()
            dp.plot(ant_sel=selection.get('ant', None),
                    time_sel=slice(index_start, index_end, 1),
                    freq_sel=selection.get('freq', None),
                    dir_sel=selection.get('dir', None),
                    pol_sel=selection.get('pol', slice(0, 1, 1)),
                    fignames=fignames,
                    observable='tec',
                    tec_eval_freq=160e6,
                    phase_wrap=True,
                    plot_facet_idx=True,
                    labels_in_radec=True,
                    solset=data_posterior)
            plt.close('all')
            if lock is not None:
                lock.release()

            screen_posterior = "screen_{}".format(posterior_name)
            fignames = [os.path.join(fig_directory, "fig-{:04d}_{}_phase.png".format(i, screen_posterior)) for i in
                        range(index_start, index_end, 1)]
            if lock is not None:
                lock.acquire()
            dp.plot(ant_sel=selection.get('ant', None),
                    time_sel=slice(index_start, index_end, 1),
                    freq_sel=selection.get('freq', None),
                    dir_sel=selection.get('dir', None),
                    pol_sel=selection.get('pol', slice(0, 1, 1)),
                    fignames=fignames,
                    observable='tec',
                    tec_eval_freq=160e6,
                    phase_wrap=True,
                    plot_facet_idx=True,
                    labels_in_radec=True,
                    solset=screen_posterior)
            plt.close('all')
            if lock is not None:
                lock.release()


            return [np.array(3).astype(np.int64)]

        return plot_results

class PlotResultsV2(Callback):
    def __init__(self, hyperparam_store, datapack, solset, index_map, lock=None, posterior_name='posterior', plot_directory='./plots', **selection):
        super(PlotResultsV2, self).__init__(hyperparam_store=hyperparam_store,
                                          lock=lock,
                                          datapack=datapack,
                                          solset=solset,
                                            index_map=index_map,
                                          posterior_name=posterior_name,
                                          plot_directory=plot_directory,
                                          **selection)

    def generate(self, hyperparam_store, datapack, solset, index_map, lock, posterior_name, plot_directory, **selection):
        self.output_dtypes = [tf.int64]
        self.name = 'PlotResultsV2'

        if not isinstance(plot_directory, str):
            raise ValueError("plot_directory should be str {}".format(type(plot_directory)))
        plot_directory = os.path.abspath(plot_directory)
        fig_directory = os.path.join(plot_directory,'phase_screens')
        # bayes_directory = os.path.join(plot_directory, 'bayes_hyperparmeters')
        os.makedirs(fig_directory,exist_ok=True)
        # os.makedirs(bayes_directory, exist_ok=True)
        dp = DatapackPlotter(datapack)

        def plot_results(index_start, index_end):
            """
            Plot results.

            :param index_start: int
                Start index of results to plot relative to 0
            :param index_end: int
                End index of results to plot relative to 0
            :return:
            """

            index_start = index_map[index_start]
            index_end = index_map[index_end-1] + 1

            data = np.load(hyperparam_store)
            keys = ['amp','y_sigma', 'lengthscales', 'a', 'b', 'timescale', 'clock_scale']
            if lock is not None:
                lock.acquire()
            fig, axs = plt.subplots(len(keys),1,sharex='all', figsize=(7,len(keys)*2))
            for i,key in enumerate(keys):
                ax = axs[i]
                ax.scatter(data['times'], data[key].flatten(),label=key)
                # ax.boxplot(data[key].T,positions=data['times'])
                # ax.set_title(key)
                ax.legend()
            plt.savefig(os.path.join(plot_directory,'hyperparameters.png'))
            plt.close('all')
            if lock is not None:
                lock.release()


            fignames = [os.path.join(fig_directory, "fig-{:04d}_{}_phase.png".format(i,solset)) for i in
                        range(index_start, index_end, 1)]
            if lock is not None:
                lock.acquire()
            dp.plot(ant_sel=selection.get('ant',None),
                    time_sel=slice(index_start, index_end, 1),
                    freq_sel=selection.get('freq',None),
                    dir_sel=selection.get('dir',None),
                    pol_sel=selection.get('pol', slice(0,1,1)),
                    fignames=fignames,
                    observable='phase',
                    phase_wrap=True,
                    plot_facet_idx=True,
                    labels_in_radec=True,
                    solset=solset)
            plt.close('all')
            if lock is not None:
                lock.release()

            # fignames = [os.path.join(fig_directory, "fig-{:04d}_{}_tec_ground_truth.png".format(i, solset)) for i in
            #             range(index_start, index_end, 1)]
            # if lock is not None:
            #     lock.acquire()
            # dp.plot(ant_sel=selection.get('ant', None),
            #         time_sel=slice(index_start, index_end, 1),
            #         freq_sel=selection.get('freq', None),
            #         dir_sel=selection.get('dir', None),
            #         pol_sel=selection.get('pol', slice(0, 1, 1)),
            #         fignames=fignames,
            #         observable='tec',
            #         tec_eval_freq=140e6,
            #         phase_wrap=True,
            #         plot_facet_idx=True,
            #         labels_in_radec=True,
            #         solset=solset)
            # plt.close('all')
            # if lock is not None:
            #     lock.release()

            # data_posterior = "data_{}".format(posterior_name)
            # fignames = [os.path.join(fig_directory, "fig-{:04d}_{}_phase.png".format(i, data_posterior)) for i in
            #             range(index_start, index_end, 1)]
            # if lock is not None:
            #     lock.acquire()
            # dp.plot(ant_sel=selection.get('ant', None),
            #         time_sel=slice(index_start, index_end, 1),
            #         freq_sel=selection.get('freq', None),
            #         dir_sel=selection.get('dir', None),
            #         pol_sel=selection.get('pol', slice(0, 1, 1)),
            #         fignames=fignames,
            #         observable='phase',
            #         phase_wrap=True,
            #         plot_facet_idx=True,
            #         labels_in_radec=True,
            #         solset=data_posterior)
            # plt.close('all')
            # if lock is not None:
            #     lock.release()

            data_posterior = "data_{}".format(posterior_name)
            fignames = [os.path.join(fig_directory, "fig-{:04d}_{}_tec_144MHz.png".format(i, data_posterior)) for i in
                        range(index_start, index_end, 1)]
            if lock is not None:
                lock.acquire()
            dp.plot(ant_sel=selection.get('ant', None),
                    time_sel=slice(index_start, index_end, 1),
                    freq_sel=selection.get('freq', None),
                    dir_sel=selection.get('dir', None),
                    pol_sel=selection.get('pol', slice(0, 1, 1)),
                    fignames=fignames,
                    observable='tec',
                    tec_eval_freq=144e6,
                    phase_wrap=True,
                    plot_facet_idx=True,
                    labels_in_radec=True,
                    solset=data_posterior)
            plt.close('all')
            if lock is not None:
                lock.release()

            # data_posterior = "data_{}".format(posterior_name)
            # fignames = [os.path.join(fig_directory, "fig-{:04d}_{}_tec_144MHz_uncert.png".format(i, data_posterior)) for i in
            #             range(index_start, index_end, 1)]
            # if lock is not None:
            #     lock.acquire()
            # dp.plot(ant_sel=selection.get('ant', None),
            #         time_sel=slice(index_start, index_end, 1),
            #         freq_sel=selection.get('freq', None),
            #         dir_sel=selection.get('dir', None),
            #         pol_sel=selection.get('pol', slice(0, 1, 1)),
            #         fignames=fignames,
            #         observable='weights_tec',
            #         tec_eval_freq=144e6,
            #         phase_wrap=False,
            #         plot_facet_idx=True,
            #         labels_in_radec=True,
            #         solset=data_posterior)
            # plt.close('all')
            # if lock is not None:
            #     lock.release()

            screen_posterior = "screen_{}".format(posterior_name)
            fignames = [os.path.join(fig_directory, "fig-{:04d}_{}_tec_144MHz.png".format(i, screen_posterior)) for i in
                        range(index_start, index_end, 1)]
            if lock is not None:
                lock.acquire()
            dp.plot(ant_sel=selection.get('ant', None),
                    time_sel=slice(index_start, index_end, 1),
                    freq_sel=selection.get('freq', None),
                    dir_sel=selection.get('dir', None),
                    pol_sel=selection.get('pol', slice(0, 1, 1)),
                    fignames=fignames,
                    observable='tec',
                    tec_eval_freq=144e6,
                    phase_wrap=True,
                    plot_facet_idx=True,
                    labels_in_radec=True,
                    solset=screen_posterior)
            plt.close('all')
            if lock is not None:
                lock.release()


            return [np.array(3).astype(np.int64)]

        return plot_results


class PlotStepsizes(Callback):
    def __init__(self, lock = None, plot_directory='./plots'):
        super(PlotStepsizes, self).__init__(
            lock=lock,
            plot_directory=plot_directory
        )

    def generate(self, lock, plot_directory):
        self.output_dtypes = [tf.int64]
        self.name = 'PlotStepsizes'

        if not isinstance(plot_directory, str):
            raise ValueError("plot_directory should be str {}".format(type(plot_directory)))
        plot_directory = os.path.join(os.path.abspath(plot_directory), 'diagnostics','stepsizes')
        os.makedirs(plot_directory, exist_ok=True)

        def plot_results(index_start, index_end, *stepsizesandlabels):
            """
            Plot the performance results.

            :param index_start: int
            :param index_end: int
            :param rhat: float array
                num_coords gelman-rubin ratio
            :param ess: float array
                num_chains, num_coords effective sample size
            :param log_accept_ratio:  float array
                num_samples, num_chains log acceptance ratio
            :param step_size: float array
                num_variables, num_samples step sizes
            :return: int
                dummy integer
            """

            N = len(stepsizesandlabels)
            if N % 2 != 0:
                raise ValueError("Must be even number of rhats and labels.")

            N = N >> 1

            stepsizes = stepsizesandlabels[:N]
            labels = stepsizesandlabels[N:]

            for label,stepsize in zip(labels, stepsizes):
                label = str(np.array(label,dtype=str))
                if lock is not None:
                    lock.acquire()
                fig, ax = plt.subplots(1,1,figsize=(4,4))
                ax.hist(stepsize,
                        bins=max(10, int(np.sqrt(stepsize.size))), label=label)
                ax.legend()
                plt.savefig(os.path.join(plot_directory,'stepsize_{}_{}_{}.png'.format(label, index_start, index_end)))
                plt.close('all')
                if lock is not None:
                    lock.release()

            return [np.array(1).astype(np.int64)]

        return plot_results


class PlotAcceptRatio(Callback):
    def __init__(self, lock=None, plot_directory='./plots'):
        super(PlotAcceptRatio, self).__init__(
                                        lock=lock,
                                          plot_directory=plot_directory,
                                          )

    def generate(self, lock, plot_directory):
        self.output_dtypes = [tf.int64]
        self.name = 'PlotAcceptRatio'

        if not isinstance(plot_directory, str):
            raise ValueError("plot_directory should be str {}".format(type(plot_directory)))
        plot_directory = os.path.join(os.path.abspath(plot_directory),'diagnostics', 'accept_ratio')
        os.makedirs(plot_directory,exist_ok=True)

        def plot_results(index_start, index_end, log_accept_ratio):
            """
            Plot the performance results.

            :param index_start: int
            :param index_end: int
            :param rhat: float array
                num_coords gelman-rubin ratio
            :param ess: float array
                num_chains, num_coords effective sample size
            :param log_accept_ratio:  float array
                num_samples, num_chains log acceptance ratio
            :param step_size: float array
                num_variables, num_samples step sizes
            :return: int
                dummy integer
            """
            if lock is not None:
                lock.acquire()
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            ax.hist(np.mean(np.exp(np.minimum(log_accept_ratio, 0.)), axis=-1),
                 bins=max(10, int(np.sqrt(log_accept_ratio.size))), label='acceptance_ratio')
            ax.legend()
            plt.savefig(os.path.join(plot_directory, 'acceptance_ratio_{}_{}.png'.format(index_start, index_end)))
            plt.close('all')
            if lock is not None:
                lock.release()

            return [np.array(1).astype(np.int64)]

        return plot_results


class PlotEss(Callback):
    def __init__(self, lock=None,plot_directory='./plots'):
        super(PlotEss, self).__init__(lock=lock,
                                          plot_directory=plot_directory,
                                          )

    def generate(self, lock, plot_directory):
        self.output_dtypes = [tf.int64]
        self.name = 'PlotEss'

        if not isinstance(plot_directory, str):
            raise ValueError("plot_directory should be str {}".format(type(plot_directory)))
        plot_directory = os.path.join(os.path.abspath(plot_directory), 'diagnostics','ess')
        os.makedirs(plot_directory,exist_ok=True)

        def plot_results(index_start, index_end, *esssandlabels):
            """
            Plot the performance results.

            :param index_start: int
            :param index_end: int
            :param rhat: float array
                num_coords gelman-rubin ratio
            :param ess: float array
                num_chains, num_coords effective sample size
            :param log_accept_ratio:  float array
                num_samples, num_chains log acceptance ratio
            :param step_size: float array
                num_variables, num_samples step sizes
            :return: int
                dummy integer
            """
            N = len(esssandlabels)
            if N%2 != 0:
                raise ValueError("Must be even number of rhats and labels.")

            N = N >> 1

            esss = esssandlabels[:N]
            labels = esssandlabels[N:]

            for label,ess in zip(labels, esss):
                label = str(np.array(label,dtype=str))
                if lock is not None:
                    lock.acquire()
                fig, ax = plt.subplots(1,1,figsize=(4,4))


                if len(ess.shape) == 2:
                    ess = np.mean(ess,axis=0)
                idx = np.arange(np.size(ess))
                ax.bar(idx, ess, label=label)
                ax.legend()
                plt.savefig(os.path.join(plot_directory,'ess_{}_{}_{}.png'.format(label, index_start, index_end)))
                plt.close('all')
                if lock is not None:
                    lock.release()

            return [np.array(1).astype(np.int64)]

        return plot_results

class PlotRhat(Callback):
    def __init__(self, lock=None, plot_directory='./plots'):
        super(PlotRhat, self).__init__(lock=lock,
                                          plot_directory=plot_directory,
                                          )

    def generate(self, lock, plot_directory):
        self.output_dtypes = [tf.int64]
        self.name = 'PlotRhat'

        if not isinstance(plot_directory, str):
            raise ValueError("plot_directory should be str {}".format(type(plot_directory)))
        plot_directory = os.path.join(os.path.abspath(plot_directory), 'diagnostics','rhat')
        os.makedirs(plot_directory,exist_ok=True)

        def plot_results(index_start, index_end, *rhatsandlabels):
            """
            Plot the performance results.

            :param index_start: int
            :param index_end: int
            :param rhat: float array
                num_coords gelman-rubin ratio
            :param ess: float array
                num_chains, num_coords effective sample size
            :param log_accept_ratio:  float array
                num_samples, num_chains log acceptance ratio
            :param step_size: float array
                num_variables, num_samples step sizes
            :return: int
                dummy integer
            """
            N = len(rhatsandlabels)
            if N%2 != 0:
                raise ValueError("Must be even number of rhats and labels.")

            N = N >> 1

            rhats = rhatsandlabels[:N]
            labels = rhatsandlabels[N:]

            for label,rhat in zip(labels, rhats):
                label = str(np.array(label,dtype=str))
                if lock is not None:
                    lock.acquire()
                fig, ax = plt.subplots(1,1,figsize=(4,4))
                idx = np.arange(np.size(rhat))
                ax.bar(idx, rhat, label=label)
                ax.set_yscale('log')
                ax.legend()
                plt.savefig(os.path.join(plot_directory,'rhat_{}_{}_{}.png'.format(label, index_start, index_end)))
                plt.close('all')
                if lock is not None:
                    lock.release()

            return [np.array(1).astype(np.int64)]

        return plot_results

class PlotELBO(Callback):
    def __init__(self, lock=None, plot_directory='./plots', index_map=None):
        super(PlotELBO, self).__init__(lock=lock,
                                       plot_directory=plot_directory,
                                       index_map=index_map
                                       )

    def generate(self, lock, plot_directory, index_map):
        self.output_dtypes = [tf.int64]
        self.name = 'PlotELBO'

        if not isinstance(plot_directory, str):
            raise ValueError("plot_directory should be str {}".format(type(plot_directory)))
        plot_directory = os.path.join(os.path.abspath(plot_directory), 'diagnostics', 'elbo')
        os.makedirs(plot_directory, exist_ok=True)

        def plot_results(index_start, index_end, elbo):
            """
            Plot the performance results.

            :param index_start: int
            :param index_end: int
            :param elbo: float array
                num_steps
            :return: int
                dummy integer
            """

            index_start = index_map[index_start]
            index_end = index_map[index_end-1] + 1

            if lock is not None:
                lock.acquire()
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            idx = np.arange(np.size(elbo))
            where_non_zero = elbo != 0.
            ax.plot(idx[where_non_zero], elbo[where_non_zero], label='elbo')
            ax.set_xlabel('iteration')
            ax.legend()
            plt.savefig(os.path.join(plot_directory, 'elbo_{:04d}_{:04d}.png'.format(index_start, index_end)))
            plt.close('all')
            if lock is not None:
                lock.release()

            return [np.array(1).astype(np.int64)]

        return plot_results


class StorePerformance(Callback):
    def __init__(self, store_file):
        super(StorePerformance, self).__init__(store_file=store_file)

    def generate(self, store_file):

        if not isinstance(store_file, str):
            raise ValueError("store_file should be str {}".format(type(store_file)))

        store_file=os.path.abspath(store_file)
        if not os.path.exists(store_file):
            np.savez(store_file, index=np.array([]), loss= [], iter_time=np.array([]), num_steps=np.array([]))


        self.output_dtypes = [tf.int64]
        self.name = 'StorePerformance'

        def store(index, loss, iter_time, num_steps):
            data = np.load(store_file)

            index = np.array([index] + list(data['index']))
            loss = np.array([np.reshape(loss,(-1,))] + list(data['loss']))
            iter_time = np.array([iter_time] + list(data['iter_time']))
            num_steps = np.array([num_steps] + list(data['num_steps']))

            np.savez(store_file,
                     index=index,
                     loss=loss,
                     iter_time=iter_time,
                     num_steps=num_steps
                     )

            return [np.array(len(index),dtype=np.int64)]

        return store