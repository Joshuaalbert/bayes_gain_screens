from scipy.interpolate import Rbf
from scipy.ndimage import median_filter
from bayes_gain_screens.misc import make_coord_array
from bayes_gain_screens import logging
from dask.multiprocessing import get
from timeit import default_timer
from graph_nets.modules import SelfAttention, GraphIndependent
from graph_nets.graphs import GraphsTuple
from graph_nets.utils_tf import _compute_stacked_offsets
import sonnet as snt
import sys

import numpy as np
import glob, os
from bayes_gain_screens.datapack import DataPack
from bayes_gain_screens.misc import voronoi_finite_polygons_2d
import matplotlib
try:
    matplotlib.use('tkagg')
except:
    pass
import pylab as plt
from scipy.spatial import Voronoi, cKDTree
from scipy.optimize import linprog
from astropy.io import fits
from astropy.wcs import WCS
import tensorflow.compat.v1 as tf
from sklearn.metrics import roc_curve

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
    band_jump = np.abs(np.diff(y, axis=-1)) > 40.
    time_flag[...,:-1][band_jump] = True
    # not sure if it was the value before of after that was bad
    time_flag[...,1:][band_jump] = True

    # # Nd, 2
    # X = (directions - np.mean(directions, axis=0)) / np.std(directions, axis=0)
    # final_flags = np.zeros_like(y, dtype=np.bool)
    # Nd, Na, Nt = y.shape
    # maxiter = Nd
    # for t in range(Nt):
    #     for a in range(Na):
    #         keep = np.logical_not(time_flag[:,a,t])
    #         if t > 0:
    #             #initialise with OR of previous flags
    #             keep = np.logical_or(keep, np.logical_not(final_flags[:, a, t-1]))
    #         if keep.sum() < Nd // 2:
    #             keep = np.ones(Nd, dtype=np.bool)
    #         for i in range(maxiter):
    #             #TODO: test solving S parallel RBF problems (sampling provides uncertainty) with smooth=0
    #             # spatially a smoothed-RBF should do a decent job of describing the TEC
    #             svm = Rbf(X[keep, 0], X[keep, 1], y[keep, a, t], smooth=0.3, function='multiquadric')
    #             y_star = svm(X[:,0], X[:,1])
    #             dy = np.abs(y_star - y[:, a, t])
    #             # print(np.median(dy), np.percentile(dy, 95))
    #             max_dy = np.max(dy[keep])
    #             if max_dy < min_res:
    #                 break
    #             # print(max_dy)
    #             keep = dy < max_dy
    #             if keep.sum() < 5:
    #                 logging.info("Possible outlier flagging divergence time: {} antenna: {}".format(t, a))
    #                 break
    #
    #         reinout_flag = reinout_filter(X[:,0], X[:,1], y[:, a, t])
    #
    #         final_flags[:, a, t] = np.logical_or(~keep, reinout_flag)
    #         # print('CONF', np.interp(dy[~keep], np.percentile(dy, np.linspace(0., 100, 100)), np.linspace(0., 100, 100)))
    #
    #         # if keep.sum() < Nd:
    #         #     logging.info("Time {} ant {} flagged {}".format(t, a, np.where(final_flags[:, a, t])))
    # final_flags = np.logical_or(final_flags, time_flag)
    final_flags = time_flag
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

def allbutone_flagger(x, y, y_std, epsilon, max_to_flag):
    '''
    Iteratively compute the exact rbf interpolant for all data, excluding a point, and compute the missing point.
    If the difference is larger than the boradcasted y_std, flag it.

    :param x: [N]
    :param y: [B,N]
    :param y_std: [B,N]
    :param epsilon: float
    :return: [B,N] np.inf where flagged, else y_std
    '''
    
    with tf.Session(graph=tf.Graph()) as sess:
        x_pl = tf.placeholder(tf.float64, x.shape)
        y_pl = tf.placeholder(tf.float64, y.shape)
        y_std_pl = tf.placeholder(tf.float64, y_std.shape)
        
        def multiquadric(x1,  x2):
            r2 = tf.math.squared_difference(x1[:, None]/epsilon,x2[None,:]/epsilon)
            return tf.math.sqrt(r2 + 1.)
        def Ab(A):
            m = tf.gather
        # N N
        A = multiquadric(x_pl, x_pl)
        N = tf.shape(A)[0]
    
        def cond(i, flags):
            # B
            done = tf.reduce_all(flags, axis=1)
            return tf.logical_and(i < max_to_flag, tf.reduce_any(tf.logical_not(done)))
    
        def body(i, cur_flags):
            """
            Predict the residual under all permutations of leave-one-out.
    
            :param i: int
            :param cur_done: tf.bool [B]
            :param cur_flags: tf.bool [B,N]
            :return:
            """
            def py_build_A(A,y, cur_flags):
                """
                Build Ab
                :param A: [B, N, N]
                :param y: [B, N]
                :param cur_flags: [B, N]
                :return:
                """
                y = y.numpy()
                A = A.numpy()
                cur_flags = cur_flags.numpy()
                ind = np.where(cur_flags)
    
                #B, N
                N = y.shape[1]
                y[cur_flags] = 0.
                yb = np.tile(y[None, :, :], [N, 1, 1])
                A[ind[0],ind[1],:] = 0.
                A[ind[0],:,ind[1]] = 0.
                A[ind[0],ind[1],ind[1]] = 1.
                #N B N N
                Ab = np.tile(A[None, :, :, :], [N, 1, 1, 1])
                #N, B, 1, N
                B = np.stack([A[:,i, :] for i in range(N)], axis=0)
    
                for i in range(N):
                    yb[i,:,i] = 0.
                    Ab[i,:,i,:] = 0.
                    Ab[i,:,:,i] = 0.
                    Ab[i,:,i,i] = 1.
                return [Ab,yb, B]
    
            #N B N N, N B N, N B 1 N
            Ab, yb, B = tf.py_function(py_build_A,[A,y_pl,cur_flags],[y_pl.dtype, y_pl.dtype, y_pl.dtype])
            #N B N 1
            wb = tf.linalg.lstsq(Ab, yb[..., None])
            #N B 1 1
            y_pred = tf.linalg.matmul(B,wb)
            #N B N
            residual = tf.math.abs(y_pl - y_pred[:,:,:,0])
            #N B N
            residual = residual*tf.cast(tf.logical_not(cur_flags), residual.dtype)
            #B N
            max_res = tf.reduce_max(residual, axis=0)
            #B N
            next_flag = max_res > y_std_pl
            #B
            # done = tf.reduce_all(next_flag, axis=1)
            return [i, next_flag]
    
        _, _, flags = tf.while_loop(cond,
                                    body,
                                    [tf.constant(0),
                                     tf.zeros(tf.shape(y_pl)[0:1], tf.bool),
                                     tf.zeros(tf.shape(y_pl), tf.bool)])
    
        flags = tf.where(flags, np.inf*tf.ones_like(y_std_pl), y_std_pl)

        return sess.run(flags, {x_pl : x, y_pl: y, y_std_pl:y_std})





# from sklearn.model_selection import StratifiedShuffleSplit

def flatten(f):
    """ Flatten a fits file so that it becomes a 2D image. Return new header and data """

    naxis = f[0].header['NAXIS']
    if naxis < 2:
        raise ValueError('Cannot make map from this')
    if naxis == 2:
        return fits.PrimaryHDU(header=f[0].header, data=f[0].data)

    w = WCS(f[0].header)
    wn = WCS(naxis=2)

    wn.wcs.crpix[0] = w.wcs.crpix[0]
    wn.wcs.crpix[1] = w.wcs.crpix[1]
    wn.wcs.cdelt = w.wcs.cdelt[0:2]
    wn.wcs.crval = w.wcs.crval[0:2]
    wn.wcs.ctype[0] = w.wcs.ctype[0]
    wn.wcs.ctype[1] = w.wcs.ctype[1]

    header = wn.to_header()
    header["NAXIS"] = 2
    copy = ('EQUINOX', 'EPOCH', 'BMAJ', 'BMIN', 'BPA', 'RESTFRQ', 'TELESCOP', 'OBSERVER')
    for k in copy:
        r = f[0].header.get(k)
        if r is not None:
            header[k] = r

    slice = []
    for i in range(naxis, 0, -1):
        if i <= 2:
            slice.append(np.s_[:], )
        else:
            slice.append(0)

    hdu = fits.PrimaryHDU(header=header, data=f[0].data[tuple(slice)])
    return hdu


def in_hull(points, x):
    n_points = len(points)
    n_dim = len(x)
    c = np.zeros(n_points)
    A = np.r_[points.T, np.ones((1, n_points))]
    b = np.r_[x, np.ones(1)]
    lp = linprog(c, A_eq=A, b_eq=b)
    return lp.success

class training_data_gen_nn(object):
    def __init__(self, K, crop_size):
        self.K = K
        self.crop_size = crop_size

    def __call__(self, label_files, ref_images, datapacks):
        for label_file, ref_image, datapack in zip(label_files, ref_images, datapacks):
            label_file, ref_image, datapack = label_file.decode(), ref_image.decode(), datapack.decode()
            print("Getting data for", label_file, ref_image, datapack, self.K)
            with fits.open(ref_image, mode='readonly') as f:
                hdu = flatten(f)
                # data = hdu.data
                wcs = WCS(hdu.header)

            tec = np.load(datapack)['tec'].copy()
            tec_uncert = np.load(datapack)['tec_uncert'].copy()
            directions = np.load(datapack)['directions'].copy()
            # dp = DataPack(datapack, readonly=True)
            # dp.current_solset = 'directionally_referenced'
            # dp.select(pol=slice(0, 1, 1))
            # tec, axes = dp.tec
            _, Nd, Na, Nt = tec.shape
            # tec_uncert, _ = dp.weights_tec
            tec_uncert = np.maximum(0.1, np.where(np.isinf(tec_uncert), 1., tec_uncert))
            # _, directions = dp.get_directions(axes['dir'])
            # directions = np.stack([directions.ra.deg, directions.dec.deg], axis=1)
            directions = wcs.wcs_world2pix(directions, 0)

            __, nn_idx = cKDTree(directions).query(directions, k=self.K + 1)

            # Nd, Na, Nt
            human_flags = np.load(label_file)
            # Nd*Na,Nt, 1
            labels = human_flags.reshape((Nd * Na, Nt, 1)).astype(np.int32)
            mask = np.reshape(human_flags != -1, (Nd * Na, Nt, 1)).astype(np.int32)
            labels = np.where(labels == -1., 0., labels)


            # tec = np.pad(tec,[(0,0),(0,0), (0,0), (window_size, window_size)],mode='reflect')
            # tec_uncert = np.pad(tec_uncert,[(0,0),(0,0), (0,0), (window_size, window_size)],mode='reflect')

            inputs = []
            for d in range(Nd):
                # K+1, Na, Nt, 2
                input = np.stack([tec[0, nn_idx[d, :], :, :] / 10., np.log(tec_uncert[0, nn_idx[d, :], :, :])], axis=-1)
                # Na, Nt, (K+1)*2
                input = np.transpose(input, (1, 2, 0, 3)).reshape((Na, Nt, (self.K + 1) * 2))
                inputs.append(input)


            # Nd*Na,Nt, (K+1)*2
            inputs = np.concatenate(inputs, axis=0)
            #buffer
            things_to_yield = []
            for b in range(inputs.shape[0]):
                if np.sum(mask[b,:,:]) == 0:
                    # print("Skipping", b)
                    continue
                # print("Reading", b)
                for start in range(0, Nt, self.crop_size):
                    stop = start + self.crop_size
                    if stop > Nt:
                        continue
                    if np.sum(mask[b,start:stop,0]) == 0:
                        continue
                    if np.random.uniform() < 0.5:
                        _yield = (inputs[b,start:stop:1,:], labels[b, start:stop:1, :], mask[b, start:stop:1, :])
                    else:
                        _yield = (inputs[b,stop:start:-1,:], labels[b, stop:start:-1, :], mask[b, stop:start:-1, :])
                    yield _yield
            #         things_to_yield.append(_yield)
            # for idx in np.random.choice(len(things_to_yield), size=len(things_to_yield), replace=False):
            #     yield things_to_yield[idx]
        return

class eval_data_gen_nn(object):
    def __init__(self, K):
        self.K = K

    def __call__(self, ref_images, datapacks):
        for ref_image, datapack in zip(ref_images, datapacks):
            ref_image, datapack = ref_image.decode(), datapack.decode()
            print("Getting data for", ref_image, datapack, self.K)
            with fits.open(ref_image, mode='readonly') as f:
                hdu = flatten(f)
                # data = hdu.data
                wcs = WCS(hdu.header)

            tec = np.load(datapack)['tec'].copy()
            tec_uncert = np.load(datapack)['tec_uncert'].copy()
            directions = np.load(datapack)['directions'].copy()
            # dp = DataPack(datapack, readonly=True)
            # dp.current_solset = 'directionally_referenced'
            # dp.select(pol=slice(0, 1, 1))
            # tec, axes = dp.tec
            _, Nd, Na, Nt = tec.shape
            # tec_uncert, _ = dp.weights_tec
            tec_uncert = np.maximum(0.1, np.where(np.isinf(tec_uncert), 1., tec_uncert))
            # _, directions = dp.get_directions(axes['dir'])
            # directions = np.stack([directions.ra.deg, directions.dec.deg], axis=1)
            directions = wcs.wcs_world2pix(directions, 0)

            __, nn_idx = cKDTree(directions).query(directions, k=self.K + 1)

            inputs = []
            for d in range(Nd):
                # K+1, Na, Nt, 2
                input = np.stack([tec[0, nn_idx[d, :], :, :] / 10., np.log(tec_uncert[0, nn_idx[d, :], :, :])], axis=-1)
                # Na, Nt, (K+1)*2
                input = np.transpose(input, (1, 2, 0, 3)).reshape((Na, Nt, (self.K + 1) * 2))
                inputs.append(input)

            # Nd*Na,Nt, (K+1)*2
            inputs = np.concatenate(inputs, axis=0)
            for b in range(inputs.shape[0]):
                yield (inputs[b,:,:],)
        return

class ClassifierNN(object):
    _module = os.path.dirname(sys.modules["bayes_gain_screens"].__file__)
    flagging_models = os.path.join(_module, 'flagging_models')
    def __init__(self, L=4, K=3, n_features=16, batch_size=16, graph=None, output_bias=0., pos_weight=1., crop_size=60):
        if graph is None:
            graph = tf.Graph()
        self.graph = graph
        self.K = K
        N = (K + 1) * 2
        self.L = L
        self.crop_size = crop_size
        self.n_features = n_features
        with self.graph.as_default():
            self.label_files_pl = tf.placeholder(tf.string, shape=[None], name='label_files')
            self.datapacks_pl = tf.placeholder(tf.string, shape=[None], name='datapacks')
            self.ref_images_pl = tf.placeholder(tf.string, shape=[None], name='ref_images')

            ###
            # train/test inputs

            dataset = tf.data.Dataset.from_tensors((self.label_files_pl, self.ref_images_pl, self.datapacks_pl))
            dataset = dataset.interleave(lambda  label_files, ref_images, datapacks:
                                         tf.data.Dataset.from_generator(
                                             training_data_gen(self.K, self.crop_size),
                                             output_types=(tf.float32, tf.int32, tf.int32),
                                             output_shapes=((self.crop_size, N),
                                                            (self.crop_size, 1),
                                                            (self.crop_size, 1),),
                                             args=(label_files, ref_images, datapacks)),
                                         cycle_length=1,
                                         block_length=1
                                         )
            train_dataset = dataset.shard(2, 0).shuffle(1000)
            train_dataset = train_dataset.batch(batch_size=batch_size, drop_remainder=True)

            test_dataset = dataset.shard(2, 1).shuffle(1000)
            test_dataset = test_dataset.batch(batch_size=batch_size, drop_remainder=True)

            iterator_tensor = train_dataset.make_initializable_iterator()
            self.train_init = iterator_tensor.initializer
            self.train_inputs, self.train_labels, self.train_mask = iterator_tensor.get_next()

            iterator_tensor = test_dataset.make_initializable_iterator()
            self.test_init = iterator_tensor.initializer
            self.test_inputs, self.test_labels, self.test_mask = iterator_tensor.get_next()

            ###
            # eval inputs
            dataset = tf.data.Dataset.from_tensors((self.ref_images_pl, self.datapacks_pl))
            dataset = dataset.interleave(lambda ref_images, datapacks:
                                         tf.data.Dataset.from_generator(
                                             eval_data_gen(self.K),
                                             output_types=(tf.float32,),
                                             output_shapes=((None, N),),
                                             args=(ref_images, datapacks)),
                                         cycle_length=1,
                                         block_length=1
                                         )
            eval_dataset = dataset.batch(batch_size=batch_size, drop_remainder=False)

            iterator_tensor = eval_dataset.make_initializable_iterator()
            self.eval_init = iterator_tensor.initializer
            self.eval_inputs, = iterator_tensor.get_next()

            ###
            # outputs

            train_outputs = self.build_model(self.train_inputs, output_bias=output_bias)
            test_outputs = self.build_model(self.test_inputs, output_bias=output_bias)
            eval_outputs = self.build_model(self.eval_inputs, output_bias=output_bias)

            num_models = len(tf.unstack(train_outputs))

            self.thresholds = tf.Variable(0.5*np.ones(num_models), shape=[num_models], dtype=tf.float32)
            self.thresholds_pl = tf.placeholder(tf.float32, [num_models])
            self.assign_thresholds = tf.assign(self.thresholds, self.thresholds_pl)

            labels_ext = tf.broadcast_to(self.train_labels, tf.shape(train_outputs))
            self.train_pred_probs = tf.nn.sigmoid(train_outputs)

            self.train_conf_mat = tf.math.confusion_matrix(tf.reshape(self.train_labels, (-1,)),
                                                           tf.reshape(
                                                               tf.reduce_mean(tf.cast(self.train_pred_probs > self.thresholds[:, None, None,
                                                                                       None], tf.float32), 0) >= 0.5, (-1,)),
                                                           weights=tf.reshape(self.train_mask, (-1,)),
                                                           num_classes=2, dtype=tf.float32)
            loss = tf.nn.weighted_cross_entropy_with_logits(labels=tf.cast(labels_ext, train_outputs.dtype), logits=train_outputs,
                                                            pos_weight=pos_weight)
            self.train_loss = tf.reduce_mean(loss * tf.cast(self.train_mask, loss.dtype))

            labels_ext = tf.broadcast_to(self.test_labels, tf.shape(test_outputs))
            self.test_pred_probs = tf.nn.sigmoid(test_outputs)
            self.test_conf_mat = tf.math.confusion_matrix(tf.reshape(self.test_labels, (-1,)),
                                                          tf.reshape(tf.reduce_mean(tf.cast(self.test_pred_probs > self.thresholds[:, None, None, None], tf.float32), 0)>=0.5, (-1,)),
                                                          weights=tf.reshape(self.test_mask, (-1,)),
                                                          num_classes=2, dtype=tf.float32)
            loss = tf.nn.weighted_cross_entropy_with_logits(labels=tf.cast(labels_ext, test_outputs.dtype), logits=test_outputs,
                                                            pos_weight=pos_weight)
            self.test_loss = tf.reduce_mean(loss * tf.cast(self.test_mask, loss.dtype))

            self.eval_pred_probs = tf.nn.sigmoid(eval_outputs)
            self.eval_pred_labels = tf.reduce_mean(tf.cast(self.eval_pred_probs > self.thresholds[:, None, None, None],
                                                          tf.float32), 0) >= 0.5

            self.global_step = tf.Variable(0, trainable=False)
            self.opt = tf.train.AdamOptimizer().minimize(self.train_loss, global_step=self.global_step)

    def conf_mat_to_str(self, conf_mat):
        tn = conf_mat[0, 0]
        fp = conf_mat[0, 1]
        fn = conf_mat[1, 0]
        tp = conf_mat[1, 1]
        T = tp + fn
        F = tn + fp
        acc = (tp + tn) / (T + F)
        rel_acc = (2. * acc - 1.) * 100.
        fpr = fp / F
        rel_fpr = (fpr / (0.5 * F / T) - 1.) * 100.
        fnr = fn / T
        rel_fnr = (fnr / (0.5 * T / F) - 1.) * 100.
        with np.printoptions(precision=2):
            return "[{} {} | {} {}] ACC: {} [{}% baseline] FPR: {} [{}% baseline] FNR: {} [{}% baseline]".format(tp, fn, fp, tn, acc, rel_acc, fpr,
                                                                                                 rel_fpr, fnr, rel_fnr)

    def train_model(self, label_files, ref_images, datapacks, epochs=10, print_freq=100, model_dir='./'):
        os.makedirs(model_dir, exist_ok=True)
        with tf.Session(graph=self.graph) as sess:
            saver = tf.train.Saver()
            print("initialising valiables")
            sess.run(tf.initialize_variables(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)))
            print("restoring if possibe")
            try:
                saver.restore(sess, tf.train.latest_checkpoint(model_dir))
            except:
                pass
            print("Running {} epochs".format(epochs))

            for epoch in range(epochs):
                print("epoch", epoch)
                print("init data for training")
                sess.run([self.train_init, self.test_init],
                         {self.label_files_pl: label_files,
                          self.ref_images_pl: ref_images,
                          self.datapacks_pl: datapacks})
                epoch_train_conf_mat = np.zeros((2, 2))
                epoch_test_conf_mat = np.zeros((2, 2))
                epoch_train_loss = []
                epoch_test_loss = []
                print("Run loop")
                epoch_test_preds = []
                epoch_test_labels = []
                epoch_test_mask = []
                batch = 0
                while True:
                    try:
                        _, train_loss, train_conf_mat, test_loss, test_conf_mat, global_step, test_preds, test_labels, \
                        test_mask = sess.run(
                            [self.opt, self.train_loss, self.train_conf_mat, self.test_loss, self.test_conf_mat,
                             self.global_step, self.test_pred_probs, self.test_labels, self.test_mask])
                        epoch_train_conf_mat += train_conf_mat
                        epoch_test_conf_mat += test_conf_mat
                        epoch_train_loss.append(train_loss)
                        epoch_test_loss.append(test_loss)
                        epoch_test_preds.append(test_preds)
                        epoch_test_labels.append(test_labels)
                        epoch_test_mask.append(test_mask)
                        if global_step % print_freq == 0:
                            with np.printoptions(precision=2):
                                print("Step {:04d} Epoch {} batch {} train loss {} test loss {}".format(global_step,
                                                                                                        epoch, batch,
                                                                                                        train_loss,
                                                                                                        test_loss))
                                print("\tTrain {}".format(self.conf_mat_to_str(train_conf_mat)))
                                print("\tTest  {}".format(self.conf_mat_to_str(test_conf_mat)))
                        batch += 1
                    except tf.errors.OutOfRangeError:
                        break
                epoch_test_preds = np.concatenate(epoch_test_preds, axis=1)
                epoch_test_labels = np.concatenate(epoch_test_labels, axis=0)
                epoch_test_labels = np.where(epoch_test_labels == -1, 0, epoch_test_labels)
                epoch_test_mask = np.concatenate(epoch_test_mask, axis=0)

                print("Epoch {} train loss: {} +- {} test loss: {} +- {}".format(epoch, np.mean(epoch_train_loss),
                                                                                 np.std(epoch_train_loss),
                                                                                 np.mean(epoch_test_loss),
                                                                                 np.std(epoch_test_loss)))
                print("Epoch {} train {} ".format(epoch, self.conf_mat_to_str(epoch_train_conf_mat)))
                print("Epoch {} test  {} ".format(epoch, self.conf_mat_to_str(epoch_test_conf_mat)))
                opt_thresholds = []
                for m in range(epoch_test_preds.shape[0]):
                    fpr, tpr, thresholds = roc_curve(epoch_test_labels.flatten(), epoch_test_preds[m,...].flatten(),
                                                     sample_weight=epoch_test_mask.flatten())
                    which =  np.argmax(tpr - fpr)
                    opt_thresholds.append(thresholds[which])
                print("New opt thresholds: {}".format(opt_thresholds))
                sess.run(self.assign_thresholds,{self.thresholds_pl:opt_thresholds})
                print('Saving...')
                save_path = saver.save(sess, self.save_path(model_dir), global_step=self.global_step)
                print("Saved to {}".format(save_path))

    def save_path(self, model_dir):
        return os.path.join(model_dir, 'model-K{:02d}-F{:02d}-L{:02d}.ckpt'.format(self.K, self.n_features, self.L))

    def get_model_file(self, model_dir):
        print("Looking in {}".format(model_dir))
        latest_model = tf.train.latest_checkpoint(model_dir)
        if latest_model is None:
            latest_model = self.save_path(model_dir)
        return latest_model

    def eval_model(self, ref_images, datapacks, model_dir=None):
        if model_dir is None:
            model_dir = self.flagging_models
        model_file = self.get_model_file(model_dir)
        with tf.Session(graph=self.graph) as sess:
            saver = tf.train.Saver()
            saver.restore(sess,model_file)
            all_predictions = []
            for ref_image, datapack in zip(ref_images, datapacks):
                sess.run(self.eval_init,
                         {self.ref_images_pl: [ref_image],
                          self.datapacks_pl: [datapack]})
                predictions = []
                while True:
                    try:
                        winner = sess.run(self.eval_pred_labels)
                        predictions.append(winner)
                    except tf.errors.OutOfRangeError:
                        break
                predictions = np.concatenate(predictions, axis=0)
                print("{} Predictions: {}".format(datapack, predictions.shape))
                print("Predicted [{}/{}] outliers ({:.2f}%) in {}".format(np.sum(predictions),predictions.size,
                                                                          100.*np.sum(predictions)/predictions.size,
                                                                          datapack))
                all_predictions.append(predictions)
            return all_predictions

    def build_model(self, inputs, output_bias=0.):
        with tf.variable_scope('classifier', reuse=tf.AUTO_REUSE):
            num = 0
            features = tf.layers.conv1d(inputs, self.n_features, [1], strides=1, padding='same', activation=None,
                                        name='conv_{:02d}'.format(num))
            num += 1
            outputs = []
            for s in [1]:
                for d in [1, 2, 3, 4]:
                    if s > 1 and d > 1:
                        continue
                    for pool in [tf.layers.average_pooling1d, tf.layers.max_pooling1d]:
                        h = [features]
                        for i in range(self.L):
                            u = tf.layers.conv1d(h[i], self.n_features, [3], s, padding='same', dilation_rate=d,
                                                 activation=tf.nn.relu,
                                                 use_bias=True, name='conv_{:02d}'.format(num))
                            num += 1
                            u = pool(u, pool_size=3, strides=1, padding='same')
                            h.append(u - h[i])
                        outputs.append(h[-1])
            # S, Nd*Na, Nt, 1
            outputs = tf.stack(
                [tf.layers.conv1d(o, 1, [1], padding='same', name='conv_{:02d}'.format(num), use_bias=False) for o in
                 outputs], axis=0)
            # outputs = tf.reduce_mean(outputs, axis=0, keepdims=True)
            output_bias = tf.Variable(output_bias, dtype=tf.float32, trainable=False)
            outputs += output_bias
            num += 1
            return outputs

def make_edges(Nd, Nt):
    node_indices = np.arange(Nd*Nt).reshape((Nd,Nt))
    senders = []
    receivers = []
    for d1 in range(Nd):
        for d2 in range(Nd):
            if d1 == d2:
                continue
            for t in range(Nt):
                senders.append(node_indices[d1,t])
                receivers.append(node_indices[d2,t])

    for t1 in range(Nt):
        for t2 in range(Nt):
            if t1 == t2:
                continue
            for d in range(Nd):
                senders.append(node_indices[d,t1])
                receivers.append(node_indices[d,t2])

    return np.array(senders, dtype=np.int32), np.array(receivers, dtype=np.int32)


class training_data_gen(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, label_files, ref_images, datapacks):
        for label_file, ref_image, datapack in zip(label_files, ref_images, datapacks):
            label_file, ref_image, datapack = label_file.decode(), ref_image.decode(), datapack.decode()
            print("Getting data for", label_file, ref_image, datapack, self.K)

            tec = np.load(datapack)['tec'].copy()
            tec_uncert = np.load(datapack)['tec_uncert'].copy()
            directions = np.load(datapack)['directions'].copy()
            _, Nd, Na, Nt = tec.shape
            tec_uncert = np.maximum(0.1, np.where(np.isinf(tec_uncert), 1., tec_uncert))

            directions -= np.mean(directions,axis=0)
            times = np.linspace(-1,1,Nt)[:, None]
            times_ext = np.tile(times[None, None, :, :], [Na, Nd, 1, 1])

            directions_ext = np.tile(directions[None, :, None, :], [Na, 1, Nt, 1])
            # Na, Nd, Nt, 2
            directions_ext = directions_ext.astype(np.float32)
            # Na, Nd, Nt, 3
            position_encoding = np.concatenate([directions_ext, times_ext], axis=-1)

            # Na, Nd, Nt
            human_flags = np.load(label_file).transpose((1,0,2))
            # Na, Nd, Nt, 1
            labels = human_flags.reshape((Na, Nd, Nt, 1)).astype(np.int32)
            mask = np.reshape(human_flags != -1, (Na , Nd, Nt, 1)).astype(np.int32)
            labels = np.where(labels == -1., 0., labels)

            #Na, Nd, Nt, 2
            inputs = np.stack([tec[0, ...] / 10., np.log(tec_uncert[0, ...])], axis=-1).transpose((1,0,2,3))

            self.crop_size = min(self.crop_size, Nt)

            senders, receivers = make_edges(Nd, self.crop_size)

            # buffer
            for b in range(inputs.shape[0]):
                # print("Reading", b)
                for start in range(0, Nt, self.crop_size):
                    stop = min(Nt,start + self.crop_size)
                    start = max(0,stop - self.crop_size)
                    if np.sum(mask[b, :, start:stop,:]) == 0:
                        # print("Skipping", b)
                        continue
                    if stop > Nt:
                        continue
                    if np.sum(mask[b, start:stop, 0]) == 0:
                        continue
                    _yield = (inputs[b, :, start:stop:1, :], labels[b, :, start:stop:1, :], mask[b, :, start:stop:1, :], position_encoding[b, :, start:stop:1,:], senders, receivers)
                    yield _yield
        return

class eval_data_gen(object):
    def __init__(self):
        pass

    def __call__(self, ref_images, datapacks):
        for ref_image, datapack in zip(ref_images, datapacks):
            ref_image, datapack = ref_image.decode(), datapack.decode()
            print("Getting data for", ref_image, datapack, self.K)

            tec = np.load(datapack)['tec'].copy()
            tec_uncert = np.load(datapack)['tec_uncert'].copy()
            directions = np.load(datapack)['directions'].copy()
            _, Nd, Na, Nt = tec.shape
            tec_uncert = np.maximum(0.1, np.where(np.isinf(tec_uncert), 1., tec_uncert))

            directions -= np.mean(directions, axis=0)
            times = np.linspace(-1, 1, Nt)[:, None]
            times_ext = np.tile(times[None, None, :, :], [Na, Nd, 1, 1])

            directions_ext = np.tile(directions[None, :, None, :], [Na, 1, Nt, 1])
            # Na, Nd, Nt, 2
            directions_ext = directions_ext.astype(np.float32)
            # Na, Nd, Nt, 3
            position_encoding = np.concatenate([directions_ext, times_ext], axis=-1)
            # Na, Nd, Nt, 2
            inputs = np.stack([tec[0, ...] / 10., np.log(tec_uncert[0, ...])], axis=-1).transpose((1, 0, 2, 3))

            senders, receivers = make_edges(Nd, Nt)

            for b in range(inputs.shape[0]):
                yield (inputs[b,:,:, :], position_encoding[b, :, :, :], position_encoding[b, :, :, :], senders, receivers)
        return



def get_output_bias(label_files):
    num_pos = 0
    num_neg = 0
    for label_file in label_files:
        human_flags = np.load(label_file)
        num_pos += np.sum(human_flags == 1)
        num_neg += np.sum(human_flags == 0)
    pos_weight = num_neg / num_pos
    bias = np.log(num_pos) - np.log(num_neg)
    return bias, pos_weight


class Classifier(object):
    _module = os.path.dirname(sys.modules["bayes_gain_screens"].__file__)
    flagging_models = os.path.join(_module, 'flagging_models')
    def __init__(self, L=4, n_features=16, batch_size=16, graph=None, output_bias=0., pos_weight=1., crop_size=60, **kwargs):
        if graph is None:
            graph = tf.Graph()
        self.graph = graph

        self.L = L
        self.crop_size = crop_size
        self.n_features = n_features
        with self.graph.as_default():
            self.label_files_pl = tf.placeholder(tf.string, shape=[None], name='label_files')
            self.datapacks_pl = tf.placeholder(tf.string, shape=[None], name='datapacks')
            self.ref_images_pl = tf.placeholder(tf.string, shape=[None], name='ref_images')
            self.shard_idx = tf.placeholder(tf.int64, shape=[], name='shard_idx')

            ###
            # train/test inputs

            dataset = tf.data.Dataset.from_tensors((self.label_files_pl, self.ref_images_pl, self.datapacks_pl))
            dataset = dataset.interleave(lambda  label_files, ref_images, datapacks:
                                         tf.data.Dataset.from_generator(
                                             training_data_gen(self.crop_size),
                                             output_types=(tf.float32, tf.int32, tf.int32, tf.float32, tf.int32, tf.int32),
                                             output_shapes=((None, self.crop_size, 2),
                                                            (None, self.crop_size, 1),
                                                            (None, self.crop_size, 1),
                                                            (None, self.crop_size, 3),
                                                            (None,),
                                                            (None,)),
                                             args=(label_files, ref_images, datapacks)),
                                         cycle_length=1,
                                         block_length=1
                                         )
            dataset = dataset.shard(2, self.shard_idx).shuffle(1000).batch(batch_size=batch_size, drop_remainder=True)

            iterator_tensor = dataset.make_initializable_iterator()
            self.init = iterator_tensor.initializer
            inputs, labels, mask, pe, senders, receivers = iterator_tensor.get_next()
            self.labels = labels
            self.mask = mask


            ###
            # eval inputs
            dataset = tf.data.Dataset.from_tensors((self.ref_images_pl, self.datapacks_pl))
            eval_dataset = dataset.interleave(lambda ref_images, datapacks:
                                         tf.data.Dataset.from_generator(
                                             eval_data_gen(),
                                             output_types=(tf.float32,tf.float32, tf.int32, tf.int32),
                                             output_shapes=((None, None, 2),
                                                            (None, None, 3),
                                                            (None,),
                                                            (None,)),
                                             args=(ref_images, datapacks)),
                                         cycle_length=1,
                                         block_length=1
                                         ).batch(batch_size=batch_size, drop_remainder=False)

            iterator_tensor = eval_dataset.make_initializable_iterator()
            self.eval_init = iterator_tensor.initializer
            eval_inputs, eval_pe, eval_senders, eval_receivers = iterator_tensor.get_next()

            ###
            # outputs
            #Nd, Nt,1
            logits = self.build_model(inputs, pe, senders, receivers, output_bias=output_bias)
            eval_logits = self.build_model(eval_inputs, eval_pe, eval_senders, eval_receivers, output_bias=output_bias)


            self.threshold = tf.Variable(0.5, shape=[], dtype=tf.float32)
            self.threshold_pl = tf.placeholder(tf.float32, [])
            self.assign_threshold = tf.assign(self.threshold, self.threshold_pl)

            pred_probs = tf.nn.sigmoid(logits)
            self.pred_probs = pred_probs
            eval_pred_probs = tf.nn.sigmoid(eval_logits)


            self.conf_mat = tf.math.confusion_matrix(labels, pred_probs > self.threshold,
                                                     weights=mask,num_classes=2,
                                                     dtype=tf.float32)

            loss = tf.nn.weighted_cross_entropy_with_logits(labels=tf.cast(labels, logits.dtype),
                                                            logits=logits,
                                                            pos_weight=pos_weight)
            self.loss = tf.reduce_mean(loss * tf.cast(mask, loss.dtype))

            self.eval_pred_labels = tf.cast(eval_pred_probs > self.threshold, tf.float32)

            self.global_step = tf.Variable(0, trainable=False)
            self.opt = tf.train.AdamOptimizer().minimize(loss, global_step=self.global_step)

    def conf_mat_to_str(self, conf_mat):
        tn = conf_mat[0, 0]
        fp = conf_mat[0, 1]
        fn = conf_mat[1, 0]
        tp = conf_mat[1, 1]
        T = tp + fn
        F = tn + fp
        acc = (tp + tn) / (T + F)
        rel_acc = (2. * acc - 1.) * 100.
        fpr = fp / F
        rel_fpr = (fpr / (0.5 * F / T) - 1.) * 100.
        fnr = fn / T
        rel_fnr = (fnr / (0.5 * T / F) - 1.) * 100.
        with np.printoptions(precision=2):
            return "ACC: {} [{}% baseline] FPR: {} [{}% baseline] FNR: {} [{}% baseline]".format(acc, rel_acc, fpr,
                                                                                                 rel_fpr, fnr, rel_fnr)

    def train_model(self, label_files, ref_images, datapacks, epochs=10, print_freq=100, model_dir='./'):
        os.makedirs(model_dir, exist_ok=True)
        with tf.Session(graph=self.graph) as sess:
            saver = tf.train.Saver()
            print("initialising valiables")
            sess.run(tf.initialize_variables(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)))
            print("restoring if possibe")
            try:
                saver.restore(sess, tf.train.latest_checkpoint(model_dir))
            except:
                pass
            print("Running {} epochs".format(epochs))

            for epoch in range(epochs):
                print("epoch", epoch)
                print("init data for training")
                sess.run([self.init],
                         {self.label_files_pl: label_files,
                          self.ref_images_pl: ref_images,
                          self.datapacks_pl: datapacks,
                          self.shard_idx: 0})

                print("Run loop")
                train_loss = 0.
                train_conf_mat = 0.
                _labels = []
                _probs = []
                _mask = []
                batch = 0
                while True:
                    try:
                        _, loss, conf_mat, global_step, pred_probs, labels, mask = sess.run(
                            [self.opt, self.loss, self.conf_mat, self.global_step, self.pred_probs, self.labels, self.mask])
                        _labels.append(labels.flatten())
                        _probs.append(pred_probs.flatten())
                        _mask.append(mask.flatten())
                        train_conf_mat += conf_mat
                        train_loss += loss

                        if global_step % print_freq == 0:
                            print("Epoch {:02d} Step {:04d} Train loss {}".format(epoch,
                                                                                  global_step,
                                                                                  loss))
                        batch += 1
                    except tf.errors.OutOfRangeError:
                        break
                train_loss /= batch
                _labels = np.concatenate(_labels)
                _probs = np.concatenate(_probs)
                _mask = np.concatenate(_mask)
                fpr, tpr, thresholds = roc_curve(_labels, _probs, sample_weight=_mask)
                which = np.argmax(tpr - fpr)
                opt_threshold = thresholds[which]
                print("New opt threshold: {}".format(opt_threshold))
                sess.run(self.assign_threshold, {self.threshold_pl: opt_threshold})

                print("init data for testing")
                sess.run([self.init],
                         {self.label_files_pl: label_files,
                          self.ref_images_pl: ref_images,
                          self.datapacks_pl: datapacks,
                          self.shard_idx: 1})
                test_loss = 0.
                test_conf_mat = 0.
                batch = 0
                while True:
                    try:
                        loss, conf_mat, pred_probs, labels, mask = sess.run(
                            [self.loss, self.conf_mat, self.pred_probs, self.labels,
                             self.mask])
                        test_conf_mat += conf_mat
                        test_loss += loss

                        batch += 1
                    except tf.errors.OutOfRangeError:
                        break
                test_loss /= batch

                print("Epoch {} Train loss: {}\nconf. mat.\n{}".format(epoch, train_loss,train_conf_mat.astype(np.int32)))
                self.conf_mat_to_str(train_conf_mat)
                print("Epoch {} Test loss: {}\nconf. mat.\n{}".format(epoch, test_loss,test_conf_mat.astype(np.int32)))
                self.conf_mat_to_str(test_conf_mat)

                print('Saving...')
                save_path = saver.save(sess, self.save_path(model_dir), global_step=self.global_step)
                print("Saved to {}".format(save_path))

    def save_path(self, model_dir):
        return os.path.join(model_dir, 'model-selfattention-F{:02d}-L{:02d}.ckpt'.format(self.n_features, self.L))

    def get_model_file(self, model_dir):
        print("Looking in {}".format(model_dir))
        latest_model = tf.train.latest_checkpoint(model_dir)
        if latest_model is None:
            latest_model = self.save_path(model_dir)
        return latest_model

    def eval_model(self, ref_images, datapacks, model_dir=None):
        if model_dir is None:
            model_dir = self.flagging_models
        model_file = self.get_model_file(model_dir)
        with tf.Session(graph=self.graph) as sess:
            saver = tf.train.Saver()
            saver.restore(sess,model_file)
            all_predictions = []
            for ref_image, datapack in zip(ref_images, datapacks):
                sess.run(self.eval_init,
                         {self.ref_images_pl: [ref_image],
                          self.datapacks_pl: [datapack]})
                predictions = []
                while True:
                    try:
                        winner = sess.run(self.eval_pred_labels)
                        predictions.append(winner)
                    except tf.errors.OutOfRangeError:
                        break
                predictions = np.concatenate(predictions, axis=0)
                print("{} Predictions: {}".format(datapack, predictions.shape))
                print("Predicted [{}/{}] outliers ({:.2f}%) in {}".format(np.sum(predictions),predictions.size,
                                                                          100.*np.sum(predictions)/predictions.size,
                                                                          datapack))
                all_predictions.append(predictions)
            return all_predictions

    def build_model(self, inputs, position_encoding, senders, receivers, output_bias=0.):
        with tf.variable_scope('classifier', reuse=tf.AUTO_REUSE):
            B = tf.shape(inputs)[0]
            Nd = tf.shape(inputs)[1]
            Nt = tf.shape(inputs)[2]
            inputs = tf.reshape(inputs, (B*Nd, Nt, 2))
            num = 0
            features = tf.layers.conv1d(inputs, self.n_features, [1], strides=1, padding='same', activation=None,
                                        name='pointwise')
            for l in range(self.L):
                features = tf.layers.conv1d(features, self.n_features, [5], strides=1, padding='same', activation=tf.nn.relu,
                                        name='conv_{:02d}'.format(l))
            features = tf.keras.layers.LayerNormalization()(features)
            features = tf.reshape(features,(B, Nd*Nt,-1))

            nodes = tf.concat([features, tf.reshape(position_encoding, (B, Nd*Nt,-1))], axis=-1)
            n_node = tf.tile(tf.shape(nodes)[1:2], [B])
            n_edge = tf.tile(tf.shape(senders)[1:2], [B])
            offsets = _compute_stacked_offsets(n_node, n_edge)
            graph = GraphsTuple(nodes=tf.reshape(nodes,(-1,)), edges=None, globals=None,n_node=n_node, n_edge=n_edge,
                                receivers=tf.reshape(receivers, (-1,))+offsets,senders=tf.reshape(senders,(-1,))+offsets)
            print(graph)
            sa1 = SelfAttention()
            gi1 = GraphIndependent(node_model_fn=snt.Sequential([snt.Linear(self.n_features), tf.nn.relu, snt.LayerNorm()]))
            graph = sa1(graph.nodes, graph.nodes, graph.nodes, graph)
            graph = gi1(graph)
            sa2 = SelfAttention()
            gi2 = GraphIndependent(node_model_fn=snt.Sequential([snt.Linear(1, use_bias=True), snt.LayerNorm()]))
            graph = sa2(graph.nodes, graph.nodes, graph.nodes, graph)
            graph = gi2(graph)
            output = tf.reshape(graph.nodes,(B, Nd, Nt, 1))

            # outputs = tf.reduce_mean(outputs, axis=0, keepdims=True)
            output_bias = tf.Variable(output_bias, dtype=tf.float32, trainable=False)
            output += output_bias
            return output


def click_through(save_file, datapack, ref_image, model_dir, model_kwargs=None):
    """
    Creates clickable app to find outliers.

    Usage:
        Typically, when you start it you have a learned model then you press 'P' to predict outliers. Otherwise a
        heuristic is used. Cyan edges show the predictions. Then typically if you trust the prediction then you press
        'c' to copy over choice. If you don't like the prediction then you left-click until the desired outliers are red
        edged. Once you find all the outliers and they are red, press 'n' to save and go to the next. You do not have to
        make non-outliers green. If a direction is not marked as an outlier (black/green/cyan) then it will be saved as
        a non-outlier. When you're done, press 'x'.

        Note: we don't reccommend using 'L' to learn a model on a single data set. You should first find many outliers
        amond all datasets, then after do a batched learning step which will do train/test split on all available data.

    Interaction:
    clicking -  left click on screen will change state (outlier/non-outlier) of facet
                right click on screen will plot the temporal evolution of the 3 closest directions (colour coded)
                    the red line show the current time-step
    keys - using the key map below.

    Colours of edges:
    cyan - predicted outliers
    black - predicted non-outlier
    red - human selected outlier
    green - human selected non-outlier

    Keymap:
    'c' - copy over the predictions [cyan]
    'n' - Go to next screen, saving current selection
    'b' - Go to previous screen, saving current selection
    'P' - run neural network prediction
    'L' - run 1 epoch learning on current dataset
    'x' - exit
    :param save_file:
    :param datapack:
    :param ref_image:
    :param model_dir:
    :param reset:
    :return:
    """
    with fits.open(ref_image) as f:
        hdu = flatten(f)
        wcs = WCS(hdu.header)
    window = 20

    dp = DataPack(datapack, readonly=True)

    dp.current_solset = 'directionally_referenced'
    dp.select(pol=slice(0, 1, 1))
    tec, axes = dp.tec
    _, Nd, Na, Nt = tec.shape
    tec_uncert, _ = dp.weights_tec
    _, directions = dp.get_directions(axes['dir'])
    directions = np.stack([directions.ra.deg, directions.dec.deg], axis=1)
    directions = wcs.wcs_world2pix(directions, 0)
    _, times = dp.get_times(axes['time'])
    times = times.mjd * 86400.
    times -= times[0]
    times /= 3600.

    window_time = times[window]

    xmin = directions[:, 0].min()
    xmax = directions[:, 0].max()
    ymin = directions[:, 1].min()
    ymax = directions[:, 1].max()

    radius = xmax - xmin

    ref_dir = directions[0:1, :]

    _, guess_flags = filter_tec_dir(tec[0, ...], directions, init_y_uncert=None, min_res=8.)
    if os.path.isfile(save_file):
        human_flags = np.load(save_file)
    else:
        human_flags = -1 * np.ones([Nd, Na, Nt], np.int32)

    guess_flags = np.where(human_flags != -1, human_flags, guess_flags)

    # compute Voronoi tesselation
    vor = Voronoi(directions)

    __, nn_idx = cKDTree(directions).query(directions, k=4)

    regions, vertices = voronoi_finite_polygons_2d(vor, radius)

    fig = plt.figure(constrained_layout=False, figsize=(12, 12))

    gs = fig.add_gridspec(3, 2)
    time_ax = fig.add_subplot(gs[0, :])
    time_ax.set_xlabel('time [hours]')
    time_ax.set_ylabel('DDTEC [mTECU]')
    vline = time_ax.plot([0, 0], [-1, 1], c='red', alpha=0.5)[0]

    time_plots = [time_ax.plot(np.arange(window * 2), 0. * np.arange(window * 2), c='black')[0] for _ in range(4)]

    dir_ax = fig.add_subplot(gs[1:, :], projection=wcs)
    dir_ax.coords[0].set_axislabel('Right Ascension (J2000)')
    dir_ax.coords[1].set_axislabel('Declination (J2000)')
    dir_ax.coords.grid(True, color='grey', ls='solid')
    polygons = []
    cmap = plt.cm.get_cmap('PuOr')
    norm = plt.Normalize(-1., 1.)
    colors = np.zeros(Nd)
    dots = []
    # colorize
    for color, region in zip(colors, regions):
        if np.size(color) == 1:
            if norm is None:
                color = cmap(color)
            else:
                color = cmap(norm(color))
        polygon = vertices[region]
        polygons.append(dir_ax.fill(*zip(*polygon), color=color, alpha=1., linewidth=4, edgecolor='black')[0])

    dir_ax.scatter(ref_dir[:, 0], ref_dir[:, 1], marker='*', color='black', zorder=19)
    for i in range(1,directions.shape[0]):
        dots.append(dir_ax.scatter(directions[i,0], directions[i,1], marker='o', s=50, c='white',ec='black', lw=2, zorder=19))

    # plt.plot(points[:,0], points[:,1], 'ko')
    dir_ax.set_xlim(vor.min_bound[0] - 0.1 * radius, vor.max_bound[0] + 0.1 * radius)
    dir_ax.set_ylim(vor.min_bound[1] - 0.1 * radius, vor.max_bound[1] + 0.1 * radius)

    def onkeyrelease(event):
        _, a, t, norm, search, order = loc
        print('Pressed {} ({}, {})'.format(event.key, event.xdata, event.ydata))
        if event.key == 'n':
            print("Saving... going to next.")
            human_flags[:, a, t] = np.where(human_flags[:, a, t] == -1, 0, human_flags[:, a, t])
            np.save(save_file, human_flags)
            next_loc = min(loc[0] + 1, len(order))
            load_data(next_loc)
        if event.key == 'b':
            print("Saving... going to back.")
            human_flags[:, a, t] = np.where(human_flags[:, a, t] == -1, 0, human_flags[:, a, t])
            np.save(save_file, human_flags)
            next_loc = max(loc[0] - 1, 0)
            load_data(next_loc)
        if event.key == 'r':
            print("Randomising order...")
            search, order = rebuild_order(random=True)
            loc[-2] = search
            loc[-1] = order
            load_data(0)
        if event.key == 'o':
            print("Ordering by guessed worst...")
            search, order = rebuild_order(random=False)
            loc[-2] = search
            loc[-1] = order
            load_data(0)
        if event.key == 'x':
            print("Exit")
            # np.save(save_file, human_flags)
            plt.close('all')
        if event.key == 'L':
            if model_kwargs is None:
                print('Model not defined.')
                return
            print("Learning one epoch")
            output_bias, pos_weight = get_output_bias([save_file])
            classifier = Classifier(L=model_kwargs.get('L'),
                                    K=model_kwargs.get('K'),
                                    n_features=model_kwargs.get('n_features'),
                                    crop_size=model_kwargs.get('crop_size'),
                                    batch_size=model_kwargs.get('batch_size'),
                                    output_bias=output_bias,
                                    pos_weight=pos_weight)
            classifier.train_model([save_file], [ref_image], [datapack.replace('.h5', '.npz')],
                                   epochs=model_kwargs.get('epochs'), print_freq=100,
                          model_dir=model_dir)
        if event.key == 'P':
            if model_kwargs is None:
                print('Model not defined.')
                return
            print("Predicting with neural net...")
            output_bias, pos_weight = 0., 1.#get_output_bias([save_file])
            classifier = Classifier(L=model_kwargs.get('L'),
                                    K=model_kwargs.get('K'),
                                    n_features=model_kwargs.get('n_features'),
                                    crop_size=model_kwargs.get('crop_size'),
                                    batch_size=model_kwargs.get('batch_size'),
                                    output_bias=output_bias,
                                    pos_weight=pos_weight)

            pred = classifier.eval_model([ref_image], [datapack.replace('.h5', '.npz')],
                                         model_dir=model_dir)[0]
            guess_flags[...] = pred.reshape((Nd, Na, Nt))
            search, order = rebuild_order()
            loc[0] = 0
            loc[4] = search
            loc[5] = order
            load_data(loc[0])

        if event.key == 'c':
            print("Copying over predicted...")
            human_flags[:, a, t] = np.where(human_flags[:, a, t] == -1, guess_flags[:, a, t], human_flags[:, a, t])
            load_data(loc[0])

    def onclick(event):
        _, a, t, norm, search, order = loc
        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              ('double' if event.dblclick else 'single', event.button,
               event.x, event.y, event.xdata, event.ydata))
        for i, region in enumerate(regions):
            point = i
            if in_hull(vertices[region], np.array([event.xdata, event.ydata])):
                print("In region {} (point {})".format(i, point))
                if event.button == 1:
                    print("Changing {}".format(human_flags[point, a, t]))
                    if human_flags[point, a, t] == -1 or human_flags[point, a, t] == 0:
                        human_flags[point, a, t] = 1
                        polygons[i].set_edgecolor('red')
                        if i > 0:
                            dots[i-1].set_edgecolor('red')
                    elif human_flags[point, a, t] == 1:
                        human_flags[point, a, t] = 0
                        polygons[i].set_edgecolor('green')
                        if i > 0:
                            dots[i-1].set_edgecolor('green')
                    polygons[i].set_zorder(11)
                    print("to {}".format(human_flags[point, a, t]))
                    scale_face()
                if event.button == 3 or event.button == 1:
                    start = max(0, t - window)
                    stop = min(Nt, t + window)
                    for n in range(4):
                        closest_idx = nn_idx[point, n]
                        time_plot = time_plots[n]
                        time_plot.set_data(times[start:stop], tec[0, closest_idx, a, start:stop])
                        if n == 0:
                            time_plot.set_color('black')
                            time_plot.set_linewidth(2)
                        else:
                            time_plot.set_color(cmap(norm(tec[0, closest_idx, a, t])))
                            time_plot.set_linewidth(1)
                    time_ax.set_xlim(times[t] - window_time, times[t] + window_time)
                    time_ax.set_ylim(tec[0, point, a, start:stop].min() - 5., tec[0, point, a, start:stop].max() + 5.)
                    vline.set_data([times[t], times[t]],
                                   [tec[0, point, a, start:stop].min() - 5., tec[0, point, a, start:stop].max() + 5.])
                fig.canvas.draw()
                # break

    cid_click = fig.canvas.mpl_connect('button_press_event', onclick)
    cid_key = fig.canvas.mpl_connect('key_release_event', onkeyrelease)

    # Na, Nt

    def rebuild_order(random=False):
        if random:
            mask = np.logical_not(np.any(human_flags >= 0, axis=0))
            search_first = np.where(mask)
            search_second = np.where(np.logical_not(mask))
            search = [list(sf) + list(ss) for sf, ss in zip(search_first, search_second)]
            order = list(np.random.choice(len(search_first[0]), len(search_first[0]), replace=False)) + \
                    list(len(search_first[0]) + np.random.choice(len(search_second[0]), len(search_second[0]),
                                                                 replace=False))
            return search, order



        mask = np.logical_and(np.any(guess_flags, axis=0), np.logical_not(np.any(human_flags>=0, axis=0)))
        search_first = np.where(mask)
        search_second = np.where(np.logical_not(mask))
        search = [list(sf) + list(ss) for sf, ss in zip(search_first, search_second)]
        order = list(np.argsort(np.sum(guess_flags==1, axis=0)[search_first])[::-1]) + \
                list(len(search_first[0]) + np.random.choice(len(search_second[0]), len(search_second[0]), replace=False))
        return search, order

    search, order = rebuild_order()
    loc = [0, 0, 0, plt.Normalize(-1., 1.), search, order]

    def scale_face():
        _, a, t, norm, search, order = loc
        vmin, vmax = np.min(tec[0, human_flags[:, a, t]<1, a, t]), np.max(tec[0, human_flags[:, a, t]<1, a, t])
        vmin, vmax = min(vmin, -vmax), max(vmax, -vmin)
        norm = plt.Normalize(vmin, vmax)
        loc[3] = norm
        for i, p in enumerate(polygons):
            p.set_facecolor(cmap(norm(tec[0, i, a, t])))

    def load_data(next_loc):
        loc[0] = next_loc
        search = loc[4]
        order = loc[5]
        o = order[next_loc]
        a = search[0][o]
        t = search[1][o]
        loc[1] = a
        loc[2] = t

        print("Looking at ant{:02d} and time {}".format(a, t))
        print("Number outliers found: {} | non-outliers {}".format(np.sum(human_flags==1), np.sum(human_flags==0)))
        scale_face()
        norm = loc[3]
        for i, p in enumerate(polygons):
            p.set_facecolor(cmap(norm(tec[0, i, a, t])))
            if human_flags[i, a, t] == 0:
                p.set_edgecolor('green')
                p.set_zorder(10)
            elif human_flags[i, a, t] == 1:
                p.set_edgecolor('red')
                p.set_zorder(11)
            elif guess_flags[i, a, t]:
                p.set_edgecolor('cyan')
                p.set_zorder(11)
            else:
                p.set_edgecolor('black')
                p.set_zorder(10)
        for i in range(len(dots)):
            dots[i].set_facecolor(cmap(norm(tec[0, i+1, a, t])))
            if human_flags[i+1, a, t] == 0:
                dots[i].set_edgecolor('green')
            elif human_flags[i+1, a, t] == 1:
                dots[i].set_edgecolor('red')
            elif guess_flags[i+1, a, t]:
                dots[i].set_edgecolor('cyan')
            else:
                dots[i].set_edgecolor('black')
        fig.canvas.draw()

    load_data(0)
    plt.show()
    return False

def remove_outliers(do_clicking, do_training, do_evaluation,
                    datapacks, ref_images, working_dir, eval_dir,
                   L=5, K=7, n_features=24, crop_size=250, batch_size=16, epochs=30):
    """

    :param do_clicking: bool, whether to interactively find outliers
    :param do_training: bool, whether to do training
    :param do_evaluation: bool, whether to evaluate models and store results.
    :param datapacks: list of datapacks (H5parms)
        Used to access 'directionally_referenced' solset to get tec/tec_uncert data.
    :param ref_images: list of images per datapack (fits)
        Used to get WCS for plotting nices and measuring distance.
    :param working_dir: str
        Where to store labels and models
    :param L: int, number layers in CNNs
    :param K: int, number of nearest neighbours to train on
    :param n_features: int, number of features to process with residual connections
    :param crop_size: int, window size to learn on
    :param batch_size: int, size of batches to learn on
    :param epochs: int, number of epochs to learn for
    :return:
    """

    print('Using working dir {}'.format(working_dir))
    working_dir = os.path.abspath(working_dir)
    os.makedirs(working_dir, exist_ok=True)
    click_dir = os.path.join(working_dir, 'click')
    os.makedirs(click_dir, exist_ok=True)
    train_dir = os.path.join(working_dir, 'train')
    os.makedirs(train_dir, exist_ok=True)

    model_kwargs = dict(L=L, K=K, n_features=n_features, crop_size=crop_size, batch_size=batch_size, epochs=epochs)

    label_files = []
    linked_datapacks = []
    linked_ref_images = []
    linked_datapack_npzs = []
    for dp, ref_img in zip(datapacks, ref_images):
        linked_datapack = os.path.join(click_dir, os.path.basename(os.path.abspath(dp)))
        if os.path.islink(linked_datapack):
            os.unlink(linked_datapack)
        print("Linking {} -> {}".format(os.path.abspath(dp), linked_datapack))
        os.symlink(os.path.abspath(dp), linked_datapack)

        linked_ref_image = linked_datapack.replace('.h5', '.ref_image.fits')
        if os.path.islink(linked_ref_image):
            os.unlink(linked_ref_image)
        print("Linking {} -> {}".format(os.path.abspath(ref_img), linked_ref_image))
        os.symlink(os.path.abspath(ref_img), linked_ref_image)

        save_file = linked_datapack.replace('.h5', '.labels.npy')
        label_files.append(save_file)
        linked_datapacks.append(linked_datapack)
        linked_ref_images.append(linked_ref_image)

        linked_datapack_npz = linked_datapack.replace('.h5', '.npz')
        if not os.path.isfile(linked_datapack_npz):
            dp = DataPack(dp, readonly=True)
            dp.current_solset = 'directionally_referenced'
            dp.select(pol=slice(0, 1, 1))
            tec, axes = dp.tec
            tec_uncert, _ = dp.weights_tec
            _, directions = dp.get_directions(axes['dir'])

            np.savez(linked_datapack_npz, tec=tec, tec_uncert=tec_uncert,
                     directions=np.stack([directions.ra.deg, directions.dec.deg], axis=1))
        linked_datapack_npzs.append(linked_datapack_npz)

    if do_clicking:
        for label_file, datapack, ref_image in zip(label_files, linked_datapacks, linked_ref_images):
            if click_through(label_file, datapack, ref_image,
                             model_dir=train_dir,
                             model_kwargs=model_kwargs):
                break

    if do_training:

        label_files = sorted(glob.glob(os.path.join(click_dir,'*.labels.npy')))
        linked_ref_images = [l.replace('.labels.npy', '.ref_image.fits') for l in label_files]
        linked_datapack_npzs = [l.replace('.labels.npy', '.npz') for l in label_files]
        print("Doing training")
        output_bias, pos_weight = get_output_bias(label_files)
        print("Output bias: {}".format(output_bias))
        print("Pos weight: {}".format(pos_weight))

        c = Classifier(L=model_kwargs.get('L'),
                       K=model_kwargs.get('K'),
                       n_features=model_kwargs.get('n_features'),
                       crop_size=model_kwargs.get('crop_size'),
                       batch_size=model_kwargs.get('batch_size'),
                       output_bias=output_bias,
                       pos_weight=pos_weight)

        c.train_model(label_files, linked_ref_images, linked_datapack_npzs, epochs=model_kwargs.get('epochs'),
                      print_freq=100,
                      model_dir=train_dir)

    if do_evaluation:
        print("Doing evaluation on all data")
        output_bias, pos_weight = 0., 1.

        c = Classifier(L=model_kwargs.get('L'),
                       K=model_kwargs.get('K'),
                       n_features=model_kwargs.get('n_features'),
                       crop_size=model_kwargs.get('crop_size'),
                       batch_size=model_kwargs.get('batch_size'),
                       output_bias=output_bias,
                       pos_weight=pos_weight)

        if eval_dir is None:
            eval_dir = train_dir

        # evaluate on what was fed in
        # linked_ref_images = sorted(glob.glob(os.path.join(click_dir, '*.ref_image.fits')))
        # linked_datapack_npzs = [l.replace('.ref_image.fits', '.npz') for l in linked_ref_images]

        predictions = c.eval_model(linked_ref_images, linked_datapack_npzs, model_dir=eval_dir)
        for i, datapack in enumerate(linked_datapacks):
            dp = DataPack(datapack, readonly=False)
            dp.current_solset = 'directionally_referenced'
            dp.select(pol=slice(0, 1, 1))
            tec_uncert, _ = dp.weights_tec
            _, Nd, Na, Nt = tec_uncert.shape
            tec_uncert = np.where(np.isinf(tec_uncert), 1., tec_uncert)
            tec_uncert = np.where(predictions[i].reshape((1, Nd, Na, Nt)) == 1, np.inf, tec_uncert)
            dp.weights_tec = tec_uncert


if __name__ == '__main__':
    datapacks = glob.glob('/home/albert/store/root_dense/L*/download_archive/L*_DDS4_full_merged.h5')
    ref_images = ['/home/albert/store/lockman/archive/image_full_ampphase_di_m.NS.app.restored.fits'] * len(datapacks)
    remove_outliers(do_clicking=True, do_training=True, do_evaluation=True,
                    datapacks=datapacks,
                    ref_images = ref_images,
                    working_dir='/home/albert/git/bayes_gain_screens/debug/outlier_detection',
                    eval_dir=None,
                    L=5, K=7, n_features=24, crop_size=250, batch_size=16, epochs=30)
    # from bayes_gain_screens.datapack import DataPack
    # dp = DataPack('/net/lofar1/data1/albert/imaging/data/lockman/L667218_DDS4_full.h5', readonly=False)
    # dp.current_solset = 'directionally_referenced'
    # select = dict(time=None, pol=slice(0,1,1))
    # dp.select(**select)
    # tec, axes = dp.tec
    # tec_uncert, _ = dp.weights_tec
    # _, directions = dp.get_directions(axes['dir'])
    # # _, freqs = dp.get_freqs(axes['freq'])
    # directions = np.stack([directions.ra.rad*np.cos(directions.dec.rad), directions.dec.rad],axis=1)
    # tec_uncert, flags = filter_tec_dir_time(tec[0,...], directions, init_y_uncert=tec_uncert[0,...], block_size=8,num_processes=64, function='multiquadric')
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
