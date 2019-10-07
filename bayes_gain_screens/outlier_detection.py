import numpy as np
from scipy.interpolate import Rbf
from scipy.ndimage import median_filter
from scipy.spatial import cKDTree
from bayes_gain_screens.misc import make_coord_array
from bayes_gain_screens import logging

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


def filter_tec_dir(y,  directions, init_y_uncert=None, **kwargs):
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
    # Nd, 2
    X = directions
    final_flags = np.zeros_like(y)
    x_list0 = list(X.T)
    Nd, Na, Nt = y.shape
    for t in range(Nt):
        for a in range(Na):
            svm = Rbf(*x_list0, y[:, a, t], smooth=0.2, **kwargs)
            y_star = svm(*x_list0)
            dy = np.abs(y_star - y[:, a, t])
            # print(np.median(dy), np.percentile(dy, 99))
            keep = dy < 55.
            # print(np.interp(55., np.percentile(dy, np.linspace(0., 100, 100)), np.linspace(0., 100, 100)))
            svm = Rbf(*list(X[keep, :].T), y[keep, a, t], smooth=0.1, **kwargs)
            y_star = svm(*x_list0)
            dy = np.abs(y_star - y[:, a, t])
            # print(np.median(dy), np.percentile(dy, 95))
            keep = dy < 15.
            # print(np.interp(20., np.percentile(dy, np.linspace(0., 100, 100)), np.linspace(0., 100, 100)))

            svm = Rbf(*list(X[keep, :].T), y[keep, a, t], smooth=0.05, **kwargs)
            y_star = svm(*x_list0)
            dy = np.abs(y_star - y[:, a, t])
            # print(np.median(dy), np.percentile(dy, 94))
            keep = dy < 10.
            # print(np.interp(15., np.percentile(dy, np.linspace(0., 100, 100)), np.linspace(0., 100, 100)))

            final_flags[:, a, t] = ~keep
            # print('CONF', np.interp(dy[~keep], np.percentile(dy, np.linspace(0., 100, 100)), np.linspace(0., 100, 100)))

            if keep.sum() < Nd:
                #9, 39, 29, 37, 11, 13
                print(t, a, np.where(final_flags[:, a, t]), 'from', Nd)

    init_y_uncert[np.where(final_flags)] = np.inf
    return init_y_uncert, final_flags

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
    directions = np.stack([directions.ra.deg, directions.dec.deg],axis=1)
    tec_uncert, flags = filter_tec_dir(tec[0,...], directions, init_y_uncert=tec_uncert[0,...], function='multiquadric')
    flags[45,...] = 1.
    tec_uncert[45, ...] = np.inf
    dp.current_solset = 'smoothed000'
    dp.select(**select)
    dp.weights_tec = tec_uncert[None,...]

    # dp.current_solset = 'sol000'
    # dp.select(time=slice(0, 40, 1), pol=slice(0, 1, 1), freq=slice(12,13,1))
    # amplitude, axes = dp.amplitude
    # _, directions = dp.get_directions(axes['dir'])
    # _, freqs = dp.get_freqs(axes['freq'])
    # directions = np.stack([directions.ra.deg, directions.dec.deg], axis=1)
    # log_amp = np.log(amplitude)
    # filter_log_amplitude_dir_freq(log_amp[0,...], directions, freqs[:, None], function='multiquadric')
