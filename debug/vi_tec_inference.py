import numpy as np
from scipy.optimize import brute, fmin
from bayes_filter import logging
from bayes_filter.datapack import DataPack
from bayes_filter.misc import make_soltab
from dask.multiprocessing import get
from collections import deque
import pymc3 as pm
import theano.tensor as tt
from pymc3.util import get_variable_name
from pymc3.distributions.continuous import get_tau_sigma
"""
This script is still being debugged/tested. 
Get's TEC from gains.
"""

class TecSolveLoss(object):
    """
    This class builds the loss function.
    Simple use case:
    # loop over data
    loss_fn = build_loss(Yreal, Yimag, freqs, gain_uncert=0.02, tec_mean_prior=0., tec_uncert_prior=0.1,S=20)
    #brute force
    tec_mean, tec_uncert = brute(loss_fn, (slice(-200, 200,1.), slice(np.log(0.01), np.log(10.), 1.), finish=fmin)
    #The results are Bayesian estimates of tec mean and uncert.

    :param Yreal: np.array shape [Nf]
        The real data (including amplitude)
    :param Yimag: np.array shape [Nf]
        The imag data (including amplitude)
    :param freqs: np.array shape [Nf]
        The freqs in Hz
    :param gain_uncert: float
        The uncertainty of gains.
    :param tec_mean_prior: float
        the prior mean for tec in mTECU
    :param tec_uncert_prior: float
        the prior tec uncert in mTECU
    :param S: int
        Number of hermite terms for Guass-Hermite quadrature
    :return: callable function of the form
        func(params) where params is a tuple or list with:
            params[0] is tec_mean in mTECU
            params[1] is log_tec_uncert in log[mTECU]
        The return of the func is a scalar loss to be minimised.
    """

    def __init__(self, Yreal, Yimag, freqs, scale_real, scale_imag, tec_mean_prior=0., tec_uncert_prior=100., S=20):
        self.x, self.w = np.polynomial.hermite.hermgauss(S)
        self.w /= np.pi

        self.tec_conv = -8.4479745e6 / freqs
        self.clock_conv = 2 * np.pi * freqs * 1e-9
        # Nf
        self.amp = np.sqrt(np.square(Yreal) + np.square(Yimag))
        self.Yreal = Yreal
        self.Yimag = Yimag

        self.scale_real = scale_real
        self.scale_imag = scale_imag

        self.tec_mean_prior = tec_mean_prior
        self.tec_uncert_prior = tec_uncert_prior

    def calculate_residuals(self, tec_mean):
        phase = tec_mean * self.tec_conv
        Yreal_m = self.amp * np.cos(phase)
        Yimag_m = self.amp * np.sin(phase)
        res_real = np.abs(self.Yreal - Yreal_m)
        res_imag = np.abs(self.Yimag - Yimag_m)

        return res_real, res_imag

    def calculate_dist_params(self, tec_mean):
        res_real, res_imag = self.calculate_residuals(tec_mean)

        scale_real = np.median(res_real)
        scale_imag = np.median(res_imag)

        return scale_real, scale_imag

    def loss_func(self, params):
        """
        VI loss
        :param params: tf.Tensor
            shapfrom scipy.optimize import brute, fmine [D]
        :return: tf.Tensor
            scalar The loss
        """
        tec_mean, log_tec_uncert = params[0], params[1]

        tec_uncert = np.exp(log_tec_uncert)

        # S
        tec = tec_mean + np.sqrt(2.) * tec_uncert * self.x
        # S, Nf
        phase = tec[:, None] * self.tec_conv
        Yreal_m = self.amp * np.cos(phase)
        Yimag_m = self.amp * np.sin(phase)
        # S
        log_prob = np.mean(-np.abs(self.Yreal - Yreal_m) / self.scale_real -
                           np.abs(self.Yimag - Yimag_m) / self.scale_imag - np.log(
            2.) - np.log(self.scale_real) - np.log(self.scale_imag), axis=-1)
        # scalar
        var_exp = np.sum(log_prob * self.w)
        # Get KL
        q_var = np.square(tec_uncert)
        tec_var_prior = np.square(self.tec_uncert_prior)
        trace = q_var / tec_var_prior
        mahalanobis = np.square(tec_mean - self.tec_mean_prior) / tec_var_prior
        constant = -1.
        logdet_qcov = np.log(tec_var_prior / q_var)
        twoKL = mahalanobis + constant + logdet_qcov + trace
        tec_prior_KL = 0.5 * twoKL
        loss = np.negative(var_exp - tec_prior_KL)
        # B
        return loss

class HeirarchicalGaussianRandomWalk(pm.distributions.distribution.Continuous):
    R"""
    Reparametrised Heirarchical Random Walk with Normal innovations

    Parameters
    ----------
    mu: tensor
        innovation drift, defaults to 0.0
    sigma : tensor
        sigma > 0, innovation standard deviation (only required if tau is not specified)
    tau : tensor
        tau > 0, innovation precision (only required if sigma is not specified)
    init : distribution
        distribution for initial value (Defaults to Flat())
    """

    def __init__(self, tau=None, init=pm.Flat.dist(), sigma=None, mu=0.,
                 sd=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if sd is not None:
            sigma = sd
        tau, sigma = get_tau_sigma(tau=tau, sigma=sigma)
        self.tau = tau = tt.as_tensor_variable(tau)
        self.sigma = self.sd = sigma = tt.as_tensor_variable(sigma)
        self.mu = mu = tt.as_tensor_variable(mu)
        self.init = init
        self.mean = tt.as_tensor_variable(0.)

    def logp(self, x):
        """
        Calculate log-probability of Gaussian Random Walk distribution at specified value.

        Parameters
        ----------
        x : numeric
            Value for which log-probability is calculated.

        Returns
        -------
        TensorVariable
        """
        tau = self.tau
        sigma = self.sigma
        mu = self.mu
        init = self.init

        x_im1 = x[:-1]
        x_i = x[1:]

        innov_like = pm.Normal.dist(mu=x_im1 + mu, sigma=sigma).logp(x_i)
        return init.logp(x[0]) + tt.sum(innov_like)

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        mu = dist.mu
        sigma = dist.sigma
        name = r'\text{%s}' % name
        return r'${} \sim \text{{GaussianRandomWalk}}(\mathit{{mu}}={},~\mathit{{sigma}}={})$'.format(name,
                                                get_variable_name(mu),
                                                get_variable_name(sigma))

def sequential_solve_random_walk(Yreal, Yimag, freqs):
    """
        Run on blocks of time.

        :param Yreal:
            [D, Nf, Nt]
        :param Yimag:
        :param freqs:
        :return:
            [D, Nt], [D, Nt]
        """

    D, Nf, N = Yreal.shape

    tec_mean_array = np.zeros((D, N))
    tec_uncert_array = np.zeros((D, N))
    neg_elbo_array = np.zeros((D, N))
    #     clock_array = np.zeros((D, N))
    #     const_array = np.zeros((D, N))
    for d in range(D):
        tec_mean_prior = 0.
        tec_uncert_prior = 150.
        # scale_real, scale_imag = 0.1, 0.1
        de_real, de_imag = deque(maxlen=30), deque(maxlen=30)
        de_real.append(0.1 * np.ones(Yreal.shape[1]))
        de_imag.append(0.1 * np.ones(Yimag.shape[1]))


        for n in range(N):
            scale_real, scale_imag = np.median(np.stack(de_real, axis=0), axis=0), np.median(np.stack(de_imag, axis=0),
                                                                                             axis=0)
            # scale_real, scale_imag = np.median(de_real[-1]), np.median(de_imag[-1])

            scale_real = np.maximum(scale_real, 0.01)
            scale_imag = np.maximum(scale_imag, 0.01)
            #             print("{} of {}, {} of {}".format(d, D, n, N))
            vi_obj = TecSolveLoss(Yreal[d, :, n], Yimag[d, :, n], freqs, scale_real, scale_imag,
                                  tec_mean_prior=tec_mean_prior, tec_uncert_prior=tec_uncert_prior)
            res0 = brute(vi_obj.loss_func,
                         (slice(-200, 200, 5),
                          slice(np.log(0.5), np.log(5.), 1)),
                         finish=fmin,
                         full_output=True)
            res_real0, res_imag0 = vi_obj.calculate_residuals(res0[0][0])

            vi_obj_check = TecSolveLoss(Yreal[d, :, n], Yimag[d, :, n], freqs, scale_real, scale_imag,
                                        tec_mean_prior=0., tec_uncert_prior=150.)
            res1 = brute(vi_obj_check.loss_func,
                         (slice(-200, 200, 5),
                          slice(np.log(0.5), np.log(5.), 1)),
                         finish=fmin, full_output=True)
            res_real1, res_imag1 = vi_obj.calculate_residuals(res1[0][0])

            if (np.sum(np.square(res_real0)) + np.sum(np.square(res_imag0))) < (
                    np.sum(np.square(res_real1)) + np.sum(np.square(res_imag1))):  # res0[1] < res1[1]:
                neg_elbo = res0[1]
                [tec_mean, log_tec_uncert] = res0[0]

            else:
                neg_elbo = res1[1]
                [tec_mean, log_tec_uncert] = res1[0]

            tec_uncert = np.exp(log_tec_uncert)
            tec_mean_prior = tec_mean
            tec_uncert_prior = 30.
            # if tec_uncert < 20.:
            #     tec_uncert_prior = 30. #np.sqrt(tec_uncert ** 2 + 30. ** 2)
            # else:
            #     tec_uncert_prior = tec_uncert #np.sqrt(tec_uncert ** 2 + 40. ** 2)
            tec_mean_array[d, n] = tec_mean
            tec_uncert_array[d, n] = tec_uncert
            neg_elbo_array[d, n] = neg_elbo

            res_real, res_imag = vi_obj.calculate_residuals(tec_mean)
            de_real.append(res_real)
            de_imag.append(res_imag)

            logging.info("{} {} {} {} {}".format(d, n, tec_mean, tec_uncert, neg_elbo))
            # new_scale_real, new_scale_imag = vi_obj.calculate_dist_params(tec_mean)
            # scale_real = scale_real * 0.9 + 0.1 * new_scale_real
            # scale_imag = scale_imag * 0.9 + 0.1 * new_scale_imag

    return tec_mean_array, tec_uncert_array, neg_elbo_array


def sequential_solve(Yreal, Yimag, freqs):
    """
    Run on blocks of time.

    :param Yreal:
        [D, Nf, Nt]
    :param Yimag:
    :param freqs:
    :return:
        [D, Nt], [D, Nt]
    """

    D, Nf, N = Yreal.shape

    tec_mean_array = np.zeros((D, N))
    tec_uncert_array = np.zeros((D, N))
    neg_elbo_array = np.zeros((D, N))
    #     clock_array = np.zeros((D, N))
    #     const_array = np.zeros((D, N))
    for d in range(D):
        tec_mean_prior = 0.
        tec_uncert_prior = 150.
        # scale_real, scale_imag = 0.1, 0.1
        de_real, de_imag = deque(maxlen=30), deque(maxlen=30)
        de_real.append(0.1 * np.ones(Yreal.shape[1]))
        de_imag.append(0.1 * np.ones(Yimag.shape[1]))
        for n in range(N):
            scale_real, scale_imag = np.median(np.stack(de_real, axis=0), axis=0), np.median(np.stack(de_imag, axis=0), axis=0)
            # scale_real, scale_imag = np.median(de_real[-1]), np.median(de_imag[-1])

            scale_real = np.maximum(scale_real, 0.01)
            scale_imag = np.maximum(scale_imag, 0.01)
            #             print("{} of {}, {} of {}".format(d, D, n, N))
            vi_obj = TecSolveLoss(Yreal[d, :, n], Yimag[d, :, n], freqs, scale_real, scale_imag,
                                  tec_mean_prior=tec_mean_prior, tec_uncert_prior=tec_uncert_prior)
            res0 = brute(vi_obj.loss_func,
                         (slice(-200, 200, 5),
                          slice(np.log(0.5), np.log(5.), 1)),
                         finish=fmin,
                         full_output=True)
            res_real0, res_imag0 = vi_obj.calculate_residuals(res0[0][0])


            vi_obj_check = TecSolveLoss(Yreal[d, :, n], Yimag[d, :, n], freqs, scale_real, scale_imag,
                                        tec_mean_prior=0., tec_uncert_prior=150.)
            res1 = brute(vi_obj_check.loss_func,
                         (slice(-200, 200, 5),
                          slice(np.log(0.5), np.log(5.), 1)),
                         finish=fmin, full_output=True)
            res_real1, res_imag1 = vi_obj.calculate_residuals(res1[0][0])

            if (np.sum(np.square(res_real0)) + np.sum(np.square(res_imag0))) < (np.sum(np.square(res_real1)) + np.sum(np.square(res_imag1))):# res0[1] < res1[1]:
                neg_elbo = res0[1]
                [tec_mean, log_tec_uncert] = res0[0]

            else:
                neg_elbo = res1[1]
                [tec_mean, log_tec_uncert] = res1[0]

            tec_uncert = np.exp(log_tec_uncert)
            tec_mean_prior = tec_mean
            tec_uncert_prior = 30.
            # if tec_uncert < 20.:
            #     tec_uncert_prior = 30. #np.sqrt(tec_uncert ** 2 + 30. ** 2)
            # else:
            #     tec_uncert_prior = tec_uncert #np.sqrt(tec_uncert ** 2 + 40. ** 2)
            tec_mean_array[d, n] = tec_mean
            tec_uncert_array[d, n] = tec_uncert
            neg_elbo_array[d, n] = neg_elbo

            res_real, res_imag = vi_obj.calculate_residuals(tec_mean)
            de_real.append(res_real)
            de_imag.append(res_imag)

            logging.info("{} {} {} {} {}".format(d, n, tec_mean, tec_uncert, neg_elbo))
            # new_scale_real, new_scale_imag = vi_obj.calculate_dist_params(tec_mean)
            # scale_real = scale_real * 0.9 + 0.1 * new_scale_real
            # scale_imag = scale_imag * 0.9 + 0.1 * new_scale_imag

    return tec_mean_array, tec_uncert_array, neg_elbo_array


def distribute_solves(datapack=None, ref_dir_idx=14, num_processes=64, numpy_data=False, elbo_save=None):
    if numpy_data:
        np_datapack = np.load(datapack)
        phase_raw, amp_raw, freqs = np_datapack['phase'], np_datapack['amplitude'], np_datapack['freqs']
        Npol, Nd, Na, Nf, Nt = phase_raw.shape
    else:
        datapack = DataPack(datapack, readonly=False)
        make_soltab(datapack, from_solset='sol000', to_solset='sol000', from_soltab='phase000', to_soltab='tec000')
        select = dict(ant=slice(1,61,3), time=slice(0,50,1), dir=None, freq=None, pol=slice(0, 1, 1))
        logging.info("Creating sol000/tec000 soltab")
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

    if ref_dir_idx is None:
        ref_dir_idx = np.argmin(
            np.median(np.reshape(np.transpose(np.abs(np.diff(phase_raw, axis=-2)), (1, 0, 2, 3, 4)), (Nd, -1)), axis=1))
        logging.info("Using ref_dir_idx {}".format(ref_dir_idx))
    phase_di = phase_raw[:, ref_dir_idx:ref_dir_idx + 1, ...]
    phase_raw = phase_raw - phase_di

    # Npol, Nd, Na, Nf, Nt
    Yimag_full = amp_raw * np.sin(phase_raw)
    Yreal_full = amp_raw * np.cos(phase_raw)

    Yimag_full = Yimag_full.reshape((-1, Nf, Nt))
    Yreal_full = Yreal_full.reshape((-1, Nf, Nt))

    D = Yimag_full.shape[0]
    dsk = {}
    for i in range(0, D, D // num_processes):
        start = i
        stop = min(i + (D // num_processes), D)
        dsk[str(i)] = (sequential_solve, Yreal_full[start:stop, :, :], Yimag_full[start:stop, :, :], freqs)
    logging.info("Running dask on {} processes".format(num_processes))
    results = get(dsk, list(dsk.keys()), num_workers=num_processes)
    logging.info("Finished dask")

    tec_mean = np.zeros((D, Nt))
    tec_uncert = np.zeros((D, Nt))
    neg_elbo = np.zeros((D, Nt))
    for c, i in enumerate(range(0, D, D // num_processes)):
        start = i
        stop = min(i + (D // num_processes), D)
        tec_mean[start:stop, :] = results[c][0]
        tec_uncert[start:stop, :] = results[c][1]
        neg_elbo[start:stop, :] = results[c][2]

    tec_mean = tec_mean.reshape((Npol, Nd, Na, Nt))
    tec_uncert = tec_uncert.reshape((Npol, Nd, Na, Nt))
    neg_elbo = neg_elbo.reshape((Npol, Nd, Na, Nt))

    tec_conv = -8.4479745e6 / freqs
    phase_model = tec_conv[:, None]*tec_mean[..., None,:]
    Yreal_model = amp_raw*np.cos(phase_model)
    Yimag_model = amp_raw*np.sin(phase_model)

    res_real = Yreal_model - Yreal_full.reshape(Yreal_model.shape)
    res_imag = Yimag_model - Yimag_full.reshape(Yimag_model.shape)
    np.savez('./residual_data.npz',res_real = res_real, res_imag = res_imag)
    print('./residual_data.npz')
    import pylab as plt
    import os
    os.makedirs('./residual_figs', exist_ok=True)
    for i in range(Na):
        for j in range(Nd):
            for b in range(0, Nt, 80):
                plt.hist(res_real[0,j, i, :, b:b+80].flatten(), bins=35)
                plt.savefig('./residual_figs/fig_{}_{}_{}.png'.format(antenna_labels[i], j, b))
                plt.close('all')


    if numpy_data:
        np.savez(datapack, phase=phase_raw, amplitude=amp_raw, freqs=freqs, tec_mean = tec_mean, tec_uncert=tec_uncert)
    else:
        logging.info("Storing results in a datapack")
        datapack.current_solset = 'sol000'
        # Npol, Nd, Na, Nf, Nt
        datapack.select(**select)
        datapack.tec = tec_mean
        datapack.weights_tec = tec_uncert
        logging.info("Stored results. Done")
    if elbo_save is not None:
        np.save(elbo_save, neg_elbo)

def test_numpy_data():
    Npol, Nd, Na, Nf, Nt = 1, 3, 1, 24, 10
    tec_true = np.random.uniform(-200, 200, size=[Npol, Nd, Na, 1]) + np.random.normal(0., 20., size=[1, 1, 1, Nt])
    freqs = np.linspace(100., 160., Nf) * 1e6
    tec_conv = -8.4479745e6 / freqs
    clock_true = 0 * np.random.normal(0., 0.4, size=[Npol, 1, Na, Nt])
    const_true = 0 * np.random.normal(0., 0.5, size=[Npol, 1, Na, Nt])
    phase_true = tec_true[..., None, :] * tec_conv[:, None] + (2. * np.pi * freqs[:, None]) * clock_true[..., None,
                                                                                              :] * 1e-9 + const_true[
                                                                                                          ..., None,
                                                                                                          :]
    amp_true = np.random.uniform(.3, 1.5, size=[Npol, Nd, Na, 1,
                                                Nt])  # + np.random.uniform(-0.1, 0.1, size=[Npol, Nd, Na, Nf, Nt])
    Yreal = amp_true * np.cos(phase_true) + np.random.laplace(scale=0.1, size=phase_true.shape)
    Yimag = amp_true * np.sin(phase_true) + np.random.laplace(scale=0.1, size=phase_true.shape)

    phase_raw = np.arctan2(Yimag, Yreal)
    amp_raw = np.sqrt(Yreal ** 2 + Yimag ** 2)

    return phase_raw, amp_raw, freqs

def transform_old_h5parm(filename, output_file):
    import tables as tb
    with tb.open_file(filename, mode='r') as f:
        print("Axes order: {}".format(f.root.sol000.phase000.val.attrs['AXES']))
        phase = f.root.sol000.phase000.val[...].transpose((4, 3, 2, 1, 0))
        amplitude = f.root.sol000.amplitude000.val[...].transpose((4, 3, 2, 1, 0))
        freqs = f.root.sol000.phase000.freq[...]
        np.savez(output_file, phase=phase, amplitude=amplitude, freqs=freqs)


if __name__ == '__main__':
    distribute_solves('/net/lofar1/data1/albert/imaging/data/P126+65_compact_raw/P126+65_full_compact_raw_v11.h5',
                       ref_dir_idx=14, num_processes=64)
    # distribute_solves('/home/albert/lofar1_1/imaging/data/LBA/zwcl0634_ddf_kms_15dir_2min.npz',#'/net/bovenrijn/data1/wwilliams/lba_test/zwcl0634/ddf_kMS_15dir/DDS5_2min_full_merged.h5',
    #                   ref_dir_idx=None, num_processes=64, numpy_data=True, elbo_save='/home/albert/lofar1_1/imaging/data/LBA/zwcl0634_ddf_kms_15dir_2min_neg_elbo.npz')
    # distribute_solves('/home/albert/lofar1_1/imaging/data/LBA/zwcl0634_ddf_kms_15dir_1min.npz',#'/net/bovenrijn/data1/wwilliams/lba_test/zwcl0634/ddf_kMS_15dir/DDS5_1min_full_merged.h5',
    #                    ref_dir_idx=None, num_processes=64, numpy_data=True, elbo_save='/home/albert/lofar1_1/imaging/data/LBA/zwcl0634_ddf_kms_15dir_1min_neg_elbo.npz')