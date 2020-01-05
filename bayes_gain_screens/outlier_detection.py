import os
os.environ['OMP_NUM_THREADS'] = "1"
import numpy as np
from scipy.interpolate import Rbf
from scipy.ndimage import median_filter
from scipy.spatial import cKDTree
from bayes_gain_screens.misc import make_coord_array, rolling_window
from bayes_gain_screens import logging
from dask.multiprocessing import get
from timeit import default_timer

def filter_const(y, init_y_uncert):
    def wrap(p):
        return np.arctan2(np.sin(p), np.cos(p))
    flags = wrap(wrap(y) - wrap(median_filter(y, (1, 1, 3)))) > 20.
    init_y_uncert[np.where(flags)] = np.inf
    return flags, init_y_uncert

def get_smooth_param(x_list, y, desired_std, **kwargs):
    smooth_ = [0.]
    std_ = [0.]
    while 10. > np.max(std_) and smooth_[-1] < 100.:
        smooth = 2.*smooth_[-1] + 0.1
        svm = Rbf(*x_list, y, smooth=smooth, **kwargs)
        y_star = svm(*x_list)
        std_.append(np.max(np.abs(y_star - y)))
        smooth_.append(smooth)
        print(smooth, std_[-1])
    return 0.1#np.interp(desired_std, std_, smooth_)


def filter_log_amplitude_dir_freq(y, directions, freqs, **kwargs):
    """
    Uses temporal and spatial smoothing.

    :param y: np.array
        [Nd, Na, Nf, Nt]
    :param y_uncert_prior: np.array
        [Nd, Na, Nf, Nt]
    :param directions: np.array
        [Nd, 2]
    :param freqs: np.array
        [Nf, 1]
    :return: np.array
        flags (1 if outlier else 0)
        [Nd, Na, Nf, Nt]
    """
    # Nd*Nf, 3
    X = make_coord_array(directions, freqs, flat=True)
    x_list0 = list(X.T)
    final_flags = np.zeros_like(y)
    Nd, Na, Nf, Nt = y.shape
    dy = np.diff(y, axis=-1)
    for t in range(Nt):
        for a in range(Na):
            if a != 55:
                continue
            print(dy[:, a, :, t])
            _y = y[:, a, :, t].reshape((-1,))
            svm = Rbf(*x_list0, _y, smooth=0.2, **kwargs)
            y_star = svm(*x_list0)
            dy = np.abs(y_star - _y)
            # print(np.median(dy), np.sort(dy)[-5:])
            keep = dy < 0.5

            svm = Rbf(*list(X[keep, :].T), _y[keep], smooth=0.1, **kwargs)
            y_star = svm(*x_list0)
            dy = np.abs(y_star - _y)
            # print(np.median(dy), np.sort(dy)[-5:])
            keep = dy < 0.3

            svm = Rbf(*list(X[keep, :].T), _y[keep], smooth=0.05, **kwargs)
            y_star = svm(*x_list0)
            dy = np.abs(y_star - _y)
            # print(np.median(dy), np.sort(dy)[-5:])
            keep = dy < 0.2
            final_flags[:, a, :, t] = ~keep.reshape((Nd, Nf))
            if keep.sum() < Nd*Nf:
                pass
                # print('CONF',
                #       np.interp(dy[~keep], np.percentile(dy, np.linspace(0., 100, 100)), np.linspace(0., 100, 100)))
                # print(t, a, np.where(final_flags[:, a, :, t]), 'from', Nd*Nf)
    return final_flags

def filter_tec(y, init_y_uncert):
    flags = (y - median_filter(y, (1, 1, 3))) > 20.
    init_y_uncert[np.where(flags)] = np.inf
    return flags, init_y_uncert
    # return filter_tec_dir(y, directions, init_y_uncert, min_res=8., function='multiquadric')

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

def reinout_filter(ra, dec, tec):
    rbftype = 'multiquadric'

    racleaned = np.copy(ra)
    deccleaned = np.copy(dec)
    TECcleaned = np.copy(tec)
    flag_idx = np.zeros(ra.size, dtype=np.bool)
    while True:
        rbf_smooth = Rbf(racleaned, deccleaned, TECcleaned, smooth=0.3, function=rbftype)
        res = np.abs(TECcleaned - rbf_smooth(racleaned, deccleaned))
        idxbadmax = np.argmax(res)
        flag_idx[idxbadmax] = True
        maxval = res[idxbadmax]
        if maxval < 8.:
            break
        else:
            idxgood = res < maxval  # keep all good
            if idxgood.sum() < 4:
                logging.info("Reinout outlier filter flagged {} of {}!".format(flag_idx.sum(), flag_idx.size))
                break
            racleaned = racleaned[idxgood]
            deccleaned = deccleaned[idxgood]
            TECcleaned = TECcleaned[idxgood]

    return flag_idx

def filter_tec_dir(y,  directions, init_y_uncert=None, min_res=8., **kwargs):
    """
    Uses temporal and spatial smoothing.

    :param y: np.array
        [Nd, Na, Nt]
    :param y_uncert_prior: np.array
        [Nd, Na, Nt]
    :param directions: np.array
        [Nd, 2]
    :return: np.array
        flags (1 if outlier else 0)
        [Nd, Na, Nt]
    """
    if init_y_uncert is None:
        init_y_uncert = 1.*np.ones_like(y)

    # mean tec over an observation should be zero for dense arrays (not for long baselines of course)
    time_flag = np.tile(np.abs(np.mean(y, axis=-1 ,keepdims=True)) > 16., [1, 1, y.shape[-1]])
    logging.info("Found [{}/{}] banding-outliers (temporal mean DDTEC > 16 mTECU)".format(time_flag.sum(), time_flag.size))
    # # jumps in tec in one time step should be less than a banding (~55 mTECU)
    # band_jump = np.abs(np.diff(y, axis=-1)) > 40.
    # time_flag[...,:-1][band_jump] = True
    # # not sure if it was the value before of after that was bad
    # time_flag[...,1:][band_jump] = True

    # Nd, 2
    X = (directions - np.mean(directions, axis=0)) / np.std(directions, axis=0)
    final_flags = np.zeros_like(y, dtype=np.bool)
    Nd, Na, Nt = y.shape
    maxiter = Nd
    for t in range(Nt):
        for a in range(Na):
            keep = np.logical_not(time_flag[:,a,t])
            if t > 0:
                #initialise with OR of previous flags
                keep = np.logical_or(keep, np.logical_not(final_flags[:, a, t-1]))
            if keep.sum() < Nd // 2:
                keep = np.ones(Nd, dtype=np.bool)
            for i in range(maxiter):
                #TODO: test solving S parallel RBF problems (sampling provides uncertainty) with smooth=0
                # spatially a smoothed-RBF should do a decent job of describing the TEC
                svm = Rbf(X[keep, 0], X[keep, 1], y[keep, a, t], smooth=0.3, **kwargs)
                y_star = svm(X[:,0], X[:,1])
                dy = np.abs(y_star - y[:, a, t])
                # print(np.median(dy), np.percentile(dy, 95))
                max_dy = np.max(dy[keep])
                if max_dy < min_res:
                    break
                # print(max_dy)
                keep = dy < max_dy
                if keep.sum() < 5:
                    logging.info("Possible outlier flagging divergence time: {} antenna: {}".format(t, a))
                    break

            reinout_flag = reinout_filter(X[:,0], X[:,1], y[:, a, t])

            final_flags[:, a, t] = np.logical_or(~keep, reinout_flag)
            # print('CONF', np.interp(dy[~keep], np.percentile(dy, np.linspace(0., 100, 100)), np.linspace(0., 100, 100)))

            # if keep.sum() < Nd:
            #     logging.info("Time {} ant {} flagged {}".format(t, a, np.where(final_flags[:, a, t])))
    final_flags = np.logical_or(final_flags, time_flag)
    logging.info("Flagged {} from {} ({:.2f}%)".format(final_flags.sum(), final_flags.size, 100. * final_flags.sum() / final_flags.size))
    final_y_uncert = np.where(final_flags, np.inf, init_y_uncert)
    return final_y_uncert, final_flags

def filter_tec_dir_time(y,  directions, init_y_uncert=None, maxiter=46, block_size=2,  num_processes=1, **kwargs):
    """
    Uses temporal and spatial smoothing.

    :param y: np.array
        [Nd, Na, Nt]
    :param y_uncert_prior: np.array
        [Nd, Na, Nt]
    :param directions: np.array
        [Nd, 2]
    :return: np.array
        flags (1 if outlier else 0)
        [Nd, Na, Nt]
    """
    if init_y_uncert is None:
        init_y_uncert = 1.*np.ones_like(y)
    if y.shape[-1] % block_size != 0:
        raise ValueError("block_size doesn't divide timesteps evenly.")
    # Nd*block_size, 3
    X = make_coord_array(directions, np.arange(block_size)[:, None], flat=True)
    X -= np.mean(X, axis=0)
    X /= np.std(X, axis=0)
    final_flags = np.zeros_like(y)
    x_list0 = list(X.T)
    Nd, Na, Nt = y.shape

    maxiter=Nd*block_size
    smooths = np.linspace(0.2, 0.3, maxiter)[::-1]
    dsk = {}
    keys = []
    for a in range(Na):
        dsk[str(a)] = (sequential_filter_tec_dir_time, Nd, Nt, X, a, block_size, kwargs,
                       maxiter, smooths, x_list0, y[:,a, :])
        keys.append(str(a))

    results = get(dsk,keys, num_workers=num_processes)
    for a in range(Na):
        final_flags[:,a,:] = results[a]
    final_uncert = np.where(final_flags, np.inf, init_y_uncert)
    return final_uncert, final_flags


def sequential_filter_tec_dir_time(Nd, Nt, X, a, block_size, kwargs, maxiter, smooths, x_list0, y):
    final_flags = np.zeros([Nd, Nt],np.bool)
    t0 = default_timer()
    for t in range(0, Nt, block_size):
        keep = np.ones(Nd * block_size, dtype=np.bool)
        for i in range(maxiter):
            y_block = y[:, t:t + block_size].reshape((-1,))
            # print(t, y_block.shape, X.shape)
            svm = Rbf(*list(X[keep, :].T), y_block[keep], smooth=smooths[i], **kwargs)
            y_star = svm(*x_list0)
            dy = np.abs(y_star - y_block)
            # print(np.median(dy), np.percentile(dy, 95))
            max_dy = np.max(dy[keep])
            if max_dy < 10.:
                break
            # print(i, max_dy)
            keep = dy < max_dy

        final_flags[:, t:t + block_size] = ~keep.reshape((Nd, block_size))

        if keep.sum() < Nd * block_size:
            # 9, 39, 29, 37, 11, 13
            print("{:.2f} per second".format(Nd*(t+block_size)/(default_timer()-t0)),
                  "block",t, "process",a,
                  "flagged", final_flags[:, t:t + block_size].sum(), 'from', Nd * block_size,
                  "({})".format(final_flags[:, t:t + block_size].sum()/(Nd*block_size)))
    return final_flags


if __name__ == '__main__':
    from bayes_gain_screens.datapack import DataPack
    dp = DataPack('/net/lofar1/data1/albert/imaging/data/lockman/L667218_DDS4_full.h5', readonly=False)
    dp.current_solset = 'directionally_referenced'
    select = dict(time=None, pol=slice(0,1,1))
    dp.select(**select)
    tec, axes = dp.tec
    tec_uncert, _ = dp.weights_tec
    _, directions = dp.get_directions(axes['dir'])
    # _, freqs = dp.get_freqs(axes['freq'])
    directions = np.stack([directions.ra.rad*np.cos(directions.dec.rad), directions.dec.rad],axis=1)
    tec_uncert, flags = filter_tec_dir_time(tec[0,...], directions, init_y_uncert=tec_uncert[0,...], block_size=8,num_processes=64, function='multiquadric')
    # flags[45,...] = 1.
    # tec_uncert[45, ...] = np.inf
    # dp.current_solset = 'smoothed000'
    # dp.select(**select)
    # dp.tec = tec
    # dp.weights_tec = tec_uncert[None,...]

    # dp.current_solset = 'sol000'
    # dp.select(time=slice(0, 40, 1), pol=slice(0, 1, 1), freq=slice(12,13,1))
    # amplitude, axes = dp.amplitude
    # _, directions = dp.get_directions(axes['dir'])
    # _, freqs = dp.get_freqs(axes['freq'])
    # directions = np.stack([directions.ra.deg, directions.dec.deg], axis=1)
    # log_amp = np.log(amplitude)
    # filter_log_amplitude_dir_freq(log_amp[0,...], directions, freqs[:, None], function='multiquadric')
