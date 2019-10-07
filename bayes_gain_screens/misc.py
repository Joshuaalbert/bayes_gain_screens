import tensorflow as tf
import numpy as np
import astropy.coordinates as ac
import astropy.units as au
import astropy.time as at
from .settings import dist_type, float_type, jitter
from timeit import default_timer
from astropy.io import fits
from matplotlib.patches import Circle
import pylab as plt
from scipy.spatial.distance import pdist
from . import logging, TEC_CONV
import os
from .datapack import DataPack
import networkx as nx
from . import logging
from collections import namedtuple
from .coord_transforms import itrs_to_enu_with_references,tf_coord_transform
from .settings import angle_type, dist_type
from scipy.special import erfinv
import datetime

def tf_datetime():
    return tf.timestamp()
    #tf.py_function(lambda: str(datetime.datetime.now()), [],[tf.string])[0]

def lock_print(lock, *message):
    if not isinstance(lock, list):
        lock = [lock]
    with tf.control_dependencies(lock):
        with tf.control_dependencies([tf.print(tf_datetime(), ":",*message)]):
            return tf.no_op()

def laplace_gaussian_marginalisation(L, y, order = 10):
    """
    Evaluate int_R^N prod_i^N L(y^i | x^i) N[x | 0, L.L^T] dx
    reparametrises as,

    int_R^M prod_i^N 1/(2 b^i) Exp[ |y^i - sqrt[2] L^i.u| - u.u ] pi^(-M/2) du


    :param L:
        shape [N, N]
    :param y:
        shape [N]
    :return:
    """
    if order > 10:
        raise ValueError("Order {} must be less than or equal to 10".format(order))

    N = tf.cast(tf.shape(y)[0], float_type)
    #N
    Ny = N*y
    #N
    halfLL = 0.5*tf.reduce_sum(tf.math.square(L), axis=-1)
    #N, N
    erf_arg = L/np.sqrt(2.) - y[:,None]/L
    # approx log(1 - erf(erf_arg))
    poly_coeffs_1 = np.array([0.,
                              -2./np.sqrt(np.pi),
                              -2./np.pi,
                              2.*(np.pi - 4)/(3.*np.pi**(3./2.)),
                              4.*(np.pi - 3.)/(3. * np.pi**2),
                              (-96. + 40.*np.pi - 3.*np.pi**2)/(15.*np.pi**(5./2.)),
                              -4.*(120. - 60.*np.pi + 7.*np.pi**2)/(45. * np.pi**3),
                              (-5760. + 3360.*np.pi - 532. * np.pi**2 + 15.*np.pi**3)/(315.*np.pi**(7./2.)),
                              8.*(-420. + 280.*np.pi - 56.*np.pi**2 + 3. * np.pi**3)/(105.*np.pi**4.),
                              (-645120. + 483840.*np.pi - 116928.*np.pi**2 + 9328.*np.pi**3. - 105. * np.pi**4)/(11340. * np.pi**(9./2.)),
                              4.*(120960. - 100800*np.pi + 28560. * np.pi**2 - 3040.*np.pi**3 + 83.*np.pi**4)/(4725. * np.pi**5)], dtype=np.float64)
    # approx log(1 + erf(erf_arg))
    poly_coeffs_2 = np.array([ -c if i%2 == 1 else c for i,c in enumerate(poly_coeffs_1)], float_type)
    #TODO: finish the function

def make_soltab(datapack:DataPack, from_solset='sol000', to_solset='sol000', from_soltab='phase000', to_soltab='tec000', select=None, directions=None, patch_names=None):
    if not isinstance(to_soltab, (list, tuple)):
        to_soltab = [to_soltab]
    if select is None:
        select = dict(ant = None, time = None, dir = None, freq = None, pol = slice(0,1,1))

    with datapack:
        datapack.current_solset = from_solset
        datapack.select(**select)
        axes = getattr(datapack, "axes_{}".format(from_soltab.replace('000', '')))
        antenna_labels, antennas = datapack.get_antennas(axes['ant'])
        if directions is None or patch_names is None:
            patch_names, directions = datapack.get_directions(axes['dir'])
        timestamps, times = datapack.get_times(axes['time'])
        freq_labels, freqs = datapack.get_freqs(axes['freq'])
        pol_labels, pols = datapack.get_pols(axes['pol'])
        Npol, Nd, Na, Nf, Nt = len(pols), len(directions), len(antennas), len(freqs), len(times)
        if to_solset not in datapack.solsets:
            datapack.add_solset(to_solset,
                                array_file=DataPack.lofar_array,
                                directions=np.stack([directions.ra.to(angle_type).value,
                                                     directions.dec.to(angle_type).value], axis=1),
                                patch_names=patch_names)
        if 'tec000' in to_soltab:
            datapack.add_soltab('tec000', weightDtype='f16', time=times.mjd * 86400., pol=pol_labels, ant=antenna_labels,
                                dir=patch_names)
        if 'clock000' in to_soltab:
            datapack.add_soltab('clock000', weightDtype='f16', time=times.mjd * 86400., pol=pol_labels,
                                ant=antenna_labels, dir=patch_names)
        if 'const000' in to_soltab:
            datapack.add_soltab('const000', weightDtype='f16', time=times.mjd * 86400., pol=pol_labels,
                                ant=antenna_labels, dir=patch_names)
        if 'phase000' in to_soltab:
            datapack.add_soltab('phase000', weightDtype='f16', freq=freqs, time=times.mjd * 86400., pol=pol_labels,
                                ant=antenna_labels, dir=patch_names)
        if 'amplitude000' in to_soltab:
            datapack.add_soltab('amplitude000', weightDtype='f16', freq=freqs, time=times.mjd * 86400., pol=pol_labels,
                                ant=antenna_labels, dir=patch_names)


def maybe_create_posterior_solsets(datapack: DataPack, solset: str, posterior_name: str,
                                   screen_directions: ac.ICRS,
                                   make_soltabs = ['phase000', 'amplitude000', 'tec000'],
                                   make_screen_solset = True, make_data_solset = True,
                                        remake_posterior_solsets=False):

    logging.info("Ensuring the solsets exist where the screen will go.")
    patch_names = ["patch_{:03d}".format(i) for i in range(len(screen_directions))]
    screen_solset = "screen_{}".format(posterior_name)
    data_solset = 'data_{}'.format(posterior_name)
    returns = []
    with datapack:
        if remake_posterior_solsets:
            if make_data_solset:
                logging.info("Deleting existing solset: {}".format(data_solset))
                datapack.delete_solset(data_solset)
            if make_screen_solset:
                logging.info("Deleting existing solset: {}".format(screen_solset))
                datapack.delete_solset(screen_solset)
        if make_data_solset:
            make_soltab(datapack, solset, data_solset, 'phase000', make_soltabs)
            returns.append(data_solset)
        if make_screen_solset:
            make_soltab(datapack, solset, screen_solset, 'phase000', make_soltabs, directions=screen_directions, patch_names=patch_names)
            returns.append(screen_solset)
        datapack.current_solset = solset
    return tuple(returns)

def dict2namedtuple(d, name="Result"):
    res = namedtuple(name, list(d.keys()))
    return res(**d)


def vertex_find(x,y):
    lhs = tf.stack([x*x, x, tf.ones_like(x)], axis=1)
    rhs = y[:, None]
    abc = tf.linalg.lstsq(lhs, rhs)
    A,B,C=abc[0,0], abc[1,0], abc[2,0]
    xmin = C - B*B/(4.*A)
    ymin = A*xmin*xmin + B*xmin + C

    argmin = tf.argmin(y)
    def_min = x[argmin]

    outside = tf.logical_or(tf.less(xmin, tf.reduce_min(x)), tf.greater(xmin, tf.reduce_max(x)))
    return tf.cond(outside, lambda: [def_min, y[argmin]], lambda: [xmin, ymin], strict=True)


# TODO: fix get set
def graph_store_set(key, value, graph = None, name="graph_store"):
    if isinstance(key,(list,tuple)):
        if len(key) != len(value):
            raise IndexError("keys and values must be equal {} {}".format(len(key), len(value)))
        for k,v in zip(key,value):
            graph_store_set(k,v,graph=graph, name=name)
    values_key = "{}_values".format(name)
    keys_key = "{}_keys".format(name)
    if graph is None:
        graph = tf.get_default_graph()
    with graph.as_default():
        graph.add_to_collection(keys_key, key)
        graph.add_to_collection(values_key, value)

def graph_store_get(key, graph = None, name="graph_store"):
    if isinstance(key, (list,tuple)):
        return [graph_store_get(k) for k in key]
    values_key = "{}_values".format(name)
    keys_key = "{}_keys".format(name)
    if graph is None:
        graph = tf.get_default_graph()
    with graph.as_default():
        keys = graph.get_collection(keys_key)
        values = graph.get_collection(values_key)
        if key not in keys:
            raise KeyError("{} not in the collection".format(key))
        index = keys_key.index(key)
        return values[index]

def plot_graph(tf_graph,ax=None, filter=False):
    '''Plot a DAG using NetworkX'''

    def children(op):
        return set(op for out in op.outputs for op in out.consumers())

    def get_graph(tf_graph):
        """Creates dictionary {node: {child1, child2, ..},..} for current
        TensorFlow graph. Result is compatible with networkx/toposort"""

        ops = tf_graph.get_operations()
        g = {}
        for op in ops:
            c = children(op)
            if len(c) == 0 and filter:
                continue
            g[op] = c
        return g

    def mapping(node):
        return node.name
    G = nx.DiGraph(get_graph(tf_graph))
    nx.relabel_nodes(G, mapping, copy=False)
    nx.draw(G, cmap = plt.get_cmap('jet'), with_labels = True, ax=ax)

def sample_exponential(shape, rate=1., dtype=float_type):
    U = tf.random.uniform(shape=shape, dtype=dtype)
    return -tf.math.log(U)/tf.convert_to_tensor(rate,dtype=dtype)

def sample_laplacian(shape, mean=0., scale=1., dtype=float_type):
    X = sample_exponential(shape, rate=1., dtype=dtype)
    Y = sample_exponential(shape, rate=1., dtype=dtype)
    return tf.convert_to_tensor(mean, dtype=dtype) +  tf.convert_to_tensor(scale, dtype=dtype) * tf.math.subtract(X,Y, name='sample_laplacian')


def log_normal_cdf_solve(x1, x2, P1=0.05, P2=0.95, as_tensor=False):
    a = np.sqrt(2.) * erfinv(2. * P1 - 1.)
    b = np.sqrt(2.) * erfinv(2. * P2 - 1.)
    log1 = np.log(x1)
    log2 = np.log(x2)
    mu = (a * log2 - b * log1) / (a - b)
    sigma = (log1 - log2) / (a - b)
    if as_tensor:
        return tf.constant(mu, float_type, name='lognormal_mu'), tf.constant(sigma, float_type, name='lognormal_sigma')
    return mu, sigma

# def make_example_datapack(Nd, Nf, Nt, pols=None,
#                           index_n=1, gain_noise=0.05,
#                           name='test.hdf5', obs_type='DDTEC',
#                           clobber=False, seed=0,
#                           square=False,
#                           kernel_hyperparams=dict(variance=1.,lengthscales=15.,b=100.,a=250., timescale=50.),
#                           return_full=False):
#     """
#
#     :param Nd:
#     :param Nf:
#     :param Nt:
#     :param pols:
#     :param time_corr:
#     :param dir_corr:
#     :param tec_scale:
#     :param tec_noise:
#     :param name:
#     :param clobber:
#     :return: DataPack
#         New object referencing a file
#     """
#     from bayes_filter.feeds import TimeFeed, IndexFeed, CoordinateFeed, init_feed
#
#     np.random.seed(seed)
#
#     logging.info("=== Creating example datapack ===")
#     name = os.path.abspath(name)
#     if os.path.isfile(name) and clobber:
#         os.unlink(name)
#
#     datapack = DataPack(name, readonly=False)
#     with datapack:
#         datapack.add_solset('sol000')
#         time0 = at.Time("2019-01-01T00:00:00.000", format='isot')
#         altaz = ac.AltAz(location=datapack.array_center.earth_location, obstime=time0)
#         up = ac.SkyCoord(alt=90.*au.deg,az=0.*au.deg,frame=altaz).transform_to('icrs')
#         directions = np.stack([np.random.normal(up.ra.rad, np.pi / 180. * 2.5, size=[Nd]),
#                                np.random.normal(up.dec.rad, np.pi / 180. * 2.5, size=[Nd])],axis=1)
#         datapack.set_directions(None,directions)
#         patch_names, _ = datapack.directions
#         _, directions = datapack.get_directions(patch_names)
#         directions = np.stack([directions.ra.to(angle_type).value, directions.dec.to(angle_type).value], axis=1)
#         antenna_labels, _ = datapack.antennas
#         _, antennas = datapack.get_antennas(antenna_labels)
#         antennas = antennas.cartesian.xyz.to(dist_type).value.T
#         # antennas /= 1000.
#         # print(directions)
#         Na = antennas.shape[0]
#
#
#         ref_dist = np.linalg.norm(antennas - antennas[0:1, :], axis=1)[None, None, :, None]  # 1,1,Na,1
#
#         times = at.Time(time0.gps+np.linspace(0, Nt * 8, Nt), format='gps').mjd[:, None] * 86400.  # mjs
#         freqs = np.linspace(120, 160, Nf) * 1e6
#         if pols is not None:
#             use_pols = True
#             assert isinstance(pols, (tuple, list))
#         else:
#             use_pols = False
#             pols = ['XX']
#         Npol = len(pols)
#         tec_conversion = TEC_CONV / freqs  # Nf
#         dtecs = []
#         with tf.Session(graph=tf.Graph()) as sess:
#             index_feed = IndexFeed(index_n)
#             time_feed = TimeFeed(index_feed, times)
#             coord_feed = CoordinateFeed(time_feed, directions, antennas, coord_map=tf_coord_transform(
#                 itrs_to_enu_with_references(antennas[0, :], [up.ra.rad, up.dec.rad], antennas[0, :])))
#             init, next = init_feed(coord_feed)
#             kern = DTECIsotropicTimeGeneral(obs_type=obs_type, kernel_params={'resolution':5}, **kernel_hyperparams)
#             K = kern.K(next)
#             Z = tf.random.normal(shape=tf.shape(K)[0:1],dtype=K.dtype)
#             ddtec = tf.matmul(safe_cholesky(K),Z[:,None])[:,0]
#             sess.run(init)
#             for t in times[::index_n]:
#                 # plt.imshow(sess.run(K))
#                 # plt.show()
#                 dtecs.append(sess.run(ddtec))
#
#         dtecs = np.concatenate(dtecs, axis=0)
#         dtecs = np.reshape(dtecs, (Nt, Nd, Na))
#         dtecs = np.transpose(dtecs, (1, 2, 0))  # Nd, Na, Nt
#         dtecs = np.tile(dtecs[None, ...], (Npol, 1, 1, 1))  # Npol, Nd, Na, Nt
#         # dtecs -= dtecs[:,:,0:1,:]
#
#         phase = dtecs[...,None,:]*tec_conversion[:,None]# Npol, Nd, Na, Nf, Nt
#         phase = np.angle(np.exp(1j*phase) + gain_noise * (np.random.laplace(size=phase.shape) + 1j*np.random.laplace(size=phase.shape)))
#
#
#         datapack.add_soltab('phase000', values=phase, ant=antenna_labels, dir = patch_names, time=times[:, 0], freq=freqs, pol=pols)
#         datapack.add_soltab('tec000', values=dtecs, ant=antenna_labels, dir = patch_names, time=times[:, 0], pol=pols)
#         if not return_full:
#             return datapack
#         return dict(datapack=datapack, directions=directions, antennas=antennas, freqs=freqs, times=times, pols=pols, dtec=dtecs, phase=phase)

def great_circle_sep(ra1, dec1, ra2, dec2):
    dra = np.abs(ra1-ra2)
    # ddec = np.abs(dec1-dec2)
    num2 = (np.cos(dec2) * np.sin(dra))**2 + (np.cos(dec1) * np.sin(dec2) - np.sin(dec1) *np.cos(dec2) * np.cos(dra))**2
    den = np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(dra)
    return np.arctan2(np.sqrt(num2),den)

def get_screen_directions(srl_fits, flux_limit = 0.1, max_N = None, min_spacing_arcmin = 1., plot=False, seed_directions=None, flux_key='Peak_flux'):
    """Given a srl file containing the sources extracted from the apparent flux image of the field,
    decide the screen directions

    :param srl_fits: str
        The path to the srl file, typically created by pybdsf
    :return: float, array [N, 2]
        The `N` sources' coordinates as an ``astropy.coordinates.ICRS`` object
    """
    logging.info("Getting screen directions from Gaussian source list.")
    hdu = fits.open(srl_fits)
    data = hdu[1].data

    arg = np.argsort(data['Total_Flux'])[::-1]

    #75MHz NLcore
    #exclude_radius = 7.82/2.


    ra = []
    dec = []
    if seed_directions is not None:
        logging.info("Using seed directions.")
        ra = list(seed_directions[:, 0])
        dec = list(seed_directions[:, 1])
    idx = []
    for i in arg:
        if data[flux_key][i] < flux_limit:
            continue
        ra_ = data['RA'][i]*np.pi/180.
        dec_ = data['DEC'][i]*np.pi/180.
        # radius = np.sqrt((ra_ - 126)**2 + (dec_ - 65)**2)
        # if radius > exclude_radius:
        #     continue
        # elif radius > 4.75/2.:
        #     high_flux = 0.5
        #     threshold = 1.
        #     f_steps = 10**np.linspace(np.log10(high_flux), np.log10(np.max(data['Total_flux'])), 1000)[::-1]
        #     f_spacing = 10**(np.linspace(np.log10(1./60.),np.log10(10./60.),1000))
        # elif radius > 3.56/2.:
        #     high_flux = 0.1
        #     threshold = 20./60.
        #     f_steps = 10**np.linspace(np.log10(high_flux), np.log10(np.max(data['Total_flux'])), 1000)[::-1]
        #     f_spacing = 10**(np.linspace(np.log10(1./60.),np.log10(10./60.),1000))
        # else:
        #     high_flux = 0.05
        #     threshold = 15./60.
        #     f_steps = 10**np.linspace(np.log10(high_flux), np.log10(np.max(data['Total_flux'])), 1000)[::-1]
        #     f_spacing = 10**(np.linspace(np.log10(1./60.),np.log10(10./60.),1000))
        # if data['Total_Flux'][i] > high_flux:
        #     a = np.searchsorted(f_steps, data['Total_Flux'][i])-1
        #     threshold = f_spacing[a]
        if len(ra) == 0:
            ra.append(ra_)
            dec.append(dec_)
            idx.append(i)
            continue
        dist = great_circle_sep(np.array(ra), np.array(dec), ra_, dec_)*180./np.pi#np.sqrt(np.square(np.subtract(ra_, ra)) + np.square(np.subtract(dec_,dec)))
        if np.all(dist > min_spacing_arcmin/60.):
            ra.append(ra_)
            dec.append(dec_)
            idx.append(i)
            continue

    f = data[flux_key][idx]
    ra = data['RA'][idx]
    dec = data['DEC'][idx]
    c = data['S_code'][idx]

    if seed_directions is not None:
        max_N -= seed_directions.shape[0]

    if max_N is not None:
        arg = np.argsort(f)[::-1][:max_N]
        f = f[arg]
        ra = ra[arg]
        dec = dec[arg]
        c = c[arg]

    if plot:
        plt.scatter(ra,dec,c=np.linspace(0.,1.,len(ra)),cmap='jet',s=np.sqrt(10000.*f),alpha=1.)

        target = Circle((np.mean(data['RA']), np.mean(data['DEC'])),radius = 3.56/2.,fc=None, alpha=0.2)
        ax = plt.gca()
        ax.add_patch(target)
        target = Circle((np.mean(data['RA']), np.mean(data['DEC'])),radius = 4.75/2.,fc=None, alpha=0.2)
        ax = plt.gca()
        ax.add_patch(target)
        plt.title("Brightest {} sources".format(len(f)))
        plt.xlabel('ra (deg)')
        plt.xlabel('dec (deg)')
        plt.savefig("screen_directions.png")
        interdist = pdist(np.stack([ra,dec],axis=1)*60.)
        plt.hist(interdist,bins=len(f))
        plt.title("inter-facet distance distribution")
        plt.xlabel('inter-facet distance [arcmin]')
        plt.savefig("interfacet_distance_dist.png")
        plt.show()

    logging.info("Found {} sources.".format(len(ra)))
    if seed_directions is not None:
        ra = list(seed_directions[:,0]) + list(ra)
        dec = list(seed_directions[:,1]) + list(dec)
    return ac.ICRS(ra=ra*au.rad, dec=dec*au.rad)


def random_sample(t, n=None):
    """
    Randomly draw `n` slices from `t`.

    :param t: float_type, tf.Tensor, [M, ...]
        Tensor to draw from
    :param n: tf.int32
        number of slices to draw. None means shuffle.
    :return: float_type, tf.Tensor, [n, ...]
        The randomly sliced tensor.
    """
    if isinstance(t, (list, tuple)):
        with tf.control_dependencies([tf.assert_equal(tf.shape(t[i])[0], tf.shape(t[0])[0]) for i in range(len(t))]):
            if n is None:
                n = tf.shape(t[0])[0]
            n = tf.minimum(n, tf.shape(t[0])[0])
            idx = tf.random_shuffle(tf.range(n))
            return [tf.gather(t[i], idx, axis=0) for i in range(len(t))]
    else:
        if n is None:
            n = tf.shape(t)[0]
        n = tf.minimum(n, tf.shape(t)[0])
        idx = tf.random_shuffle(tf.range(n))
        return tf.gather(t, idx, axis=0)

def K_parts(kern, *X):
    L = len(X)
    K = [[None for _ in range(L)] for _ in range(L)]
    for i in range(L):
        for j in range(i,L):
            K_part = kern.matrix(X[i], X[j])
            K[i][j] = K_part
            if i == j:
                continue
            K[j][i] = tf.transpose(K_part,(1,0) if kern.squeeze else (0,2,1))
        K[i] = tf.concat(K[i],axis=-1)
    K = tf.concat(K, axis=-2)
    return K

def log_normal_solve_fwhm(a,b,D=0.5):
    """Solve the parameters for a log-normal distribution given the 1/D power at limits a and b.

    :param a: float
        The lower D power
    :param b: float
        The upper D power
    :return: tuple of (mu, stddev) parametrising the log-normal distribution
    """
    if b < a:
        raise ValueError("b should be greater than a")
    lower = np.log(a)
    upper = np.log(b)
    d = upper - lower #2 sqrt(2 sigma**2 ln(1/D))
    sigma2 = 0.5*(0.5*d)**2/np.log(1./D)
    s = upper + lower #2 (mu - sigma**2)
    mu = 0.5*s + sigma2
    return np.array(mu,dtype=np.float64), np.array(np.sqrt(sigma2),dtype=np.float64)



def diagonal_jitter(N, _jitter=None):
    """
    Create diagonal matrix with jitter on the diagonal

    :param N: int, tf.int32
        The size of diagonal
    :return: float_type, Tensor, [N, N]
        The diagonal matrix with jitter on the diagonal.
    """
    if _jitter is None:
        _jitter = jitter
    return tf.linalg.tensor_diag(tf.fill([N],tf.convert_to_tensor(_jitter,float_type)))

def safe_cholesky(K, _jitter=None):
    n = tf.shape(K)[-1]
    s = tf.reduce_mean(tf.linalg.diag_part(K),axis=-1, keep_dims=True)[..., None]
    K = K/s
    L = tf.sqrt(s)*tf.linalg.cholesky(K + diagonal_jitter(n, _jitter=_jitter))
    # else:
    #     s = tf.linalg.diag_part(K)#[b1,...,bB, N]
    #     # K_ij = s_ik L_kl L_pl s_pj
    #     K = K / (s[...,None]*s[...,None,:])
    #     L = s[...,None] * tf.linalg.cholesky(K + diagonal_jitter(n, _jitter=_jitter))

    return L

def timer():
    """
    Return system time as tensorflow op
    :return:
    """
    return tf.cast(tf.timestamp(), float_type)
    #tf.py_function(default_timer, [], float_type)

def flatten_batch_dims(t, num_batch_dims=None):
    """
    Flattens the first `num_batch_dims`
    :param t: Tensor [b0,...bB, n0,...nN]
        Flattening happens for first `B` dimensions
    :param num_batch_dims: int, or tf.int32
        Number of dims in batch to flatten. If None then all but last. If < 0 then count from end.
    :return: Tensor [b0*...*bB, n0,...,nN]
    """
    shape = tf.shape(t)
    if num_batch_dims is None:
        num_batch_dims =  - 1
    out_shape = tf.concat([[-1], shape[num_batch_dims:]],axis=0)
    return tf.reshape(t,out_shape)

def make_coord_array(*X, flat=True, coord_map=None):
    """
    Create the design matrix from a list of coordinates
    :param X: list of length p of float, array [Ni, D]
        Ni can be different for each coordinate array, but D must be the same.
    :param flat: bool
        Whether to return a flattened representation
    :param coord_map: callable(coordinates), optional
            If not None then get mapped over the coordinates
    :return: float, array [N0,...,Np, D] if flat=False else [N0*...*Np, D]
        The coordinate design matrix
    """

    if coord_map is not None:
        X = [coord_map(x) for x in X]

    def add_dims(x, where, sizes):
        shape = []
        tiles = []
        for i in range(len(sizes)):
            if i not in where:
                shape.append(1)
                tiles.append(sizes[i])
            else:
                shape.append(-1)
                tiles.append(1)
        return np.tile(np.reshape(x, shape), tiles)

    N = [x.shape[0] for x in X]
    X_ = []

    for i, x in enumerate(X):
        for dim in range(x.shape[1]):
            X_.append(add_dims(x[:, dim], [i], N))
    X = np.stack(X_, axis=-1)
    if not flat:
        return X
    return np.reshape(X, (-1, X.shape[-1]))


def load_array_file(array_file):
    '''Loads a csv where each row is x,y,z in geocentric ITRS coords of the antennas'''

    try:
        types = np.dtype({'names': ['X', 'Y', 'Z', 'diameter', 'station_label'],
                          'formats': [np.double, np.double, np.double, np.double, 'S16']})
        d = np.genfromtxt(array_file, comments='#', dtype=types)
        diameters = d['diameter']
        labels = np.array(d['station_label'].astype(str))
        locs = ac.SkyCoord(x=d['X'] * au.m, y=d['Y'] * au.m, z=d['Z'] * au.m, frame='itrs')
        Nantenna = int(np.size(d['X']))
    except:
        d = np.genfromtxt(array_file, comments='#', usecols=(0, 1, 2))
        locs = ac.SkyCoord(x=d[:, 0] * au.m, y=d[:, 1] * au.m, z=d[:, 2] * au.m, frame='itrs')
        Nantenna = d.shape[0]
        labels = np.array(["ant{:02d}".format(i) for i in range(Nantenna)])
        diameters = None
    return np.array(labels).astype(np.str_), locs.cartesian.xyz.to(dist_type).value.transpose()

# def save_array_file(array_file):
#     import time
#     ants = _solset.getAnt()
#     labels = []
#     locs = []
#     for label, pos in ants.items():
#         labels.append(label)
#         locs.append(pos)
#     Na = len(labels)
# with open(array_file, 'w') as f:
#     f.write('# Created on {0} by Joshua G. Albert\n'.format(time.strftime("%a %c", time.localtime())))
#     f.write('# ITRS(m)\n')
#     f.write('# X\tY\tZ\tlabels\n')
#     i = 0
#     while i < Na:
#         f.write(
#             '{0:1.9e}\t{1:1.9e}\t{2:1.9e}\t{3:d}\t{4}'.format(locs[i][0], locs[i][1], locs[i][2], labels[i]))
#         if i < Na - 1:
#             f.write('\n')
#         i += 1


@tf.custom_gradient
def sqrt_with_finite_grads(x, name=None):
    """A sqrt function whose gradient at zero is very large but finite.

    Args:
    x: a `Tensor` whose sqrt is to be computed.
    name: a Python `str` prefixed to all ops created by this function.
      Default `None` (i.e., "sqrt_with_finite_grads").

    Returns:
    sqrt: the square root of `x`, with an overridden gradient at zero
    grad: a gradient function, which is the same as sqrt's gradient everywhere
      except at zero, where it is given a large finite value, instead of `inf`.

    Raises:
    TypeError: if `tf.convert_to_tensor(x)` is not a `float` type.

    Often in kernel functions, we need to compute the L2 norm of the difference
    between two vectors, `x` and `y`: `sqrt(sum_i((x_i - y_i) ** 2))`. In the
    case where `x` and `y` are identical, e.g., on the diagonal of a kernel
    matrix, we get `NaN`s when we take gradients with respect to the inputs. To
    see, this consider the forward pass:

    ```
    [x_1 ... x_N]  -->  [x_1 ** 2 ... x_N ** 2]  -->
        (x_1 ** 2 + ... + x_N ** 2)  -->  sqrt((x_1 ** 2 + ... + x_N ** 2))
    ```

    When we backprop through this forward pass, the `sqrt` yields an `inf` because
    `grad_z(sqrt(z)) = 1 / (2 * sqrt(z))`. Continuing the backprop to the left, at
    the `x ** 2` term, we pick up a `2 * x`, and when `x` is zero, we get
    `0 * inf`, which is `NaN`.

    We'd like to avoid these `NaN`s, since they infect the rest of the connected
    computation graph. Practically, when two inputs to a kernel function are
    equal, we are in one of two scenarios:
    1. We are actually computing k(x, x), in which case norm(x - x) is
       identically zero, independent of x. In this case, we'd like the
       gradient to reflect this independence: it should be zero.
    2. We are computing k(x, y), and x just *happens* to have the same value
       as y. The gradient at such inputs is in fact ill-defined (there is a
       cusp in the sqrt((x - y) ** 2) surface along the line x = y). There are,
       however, an infinite number of sub-gradients, all of which are valid at
       all such inputs. By symmetry, there is exactly one which is "special":
       zero, and we elect to use that value here. In practice, having two
       identical inputs to a kernel matrix is probably a pathological
       situation to be avoided, but that is better resolved at a higher level
       than this.

    To avoid the infinite gradient at zero, we use tf.custom_gradient to redefine
    the gradient at zero. We assign it to be a very large value, specifically
    the sqrt of the max value of the floating point dtype of the input. We use
    the sqrt (as opposed to just using the max floating point value) to avoid
    potential overflow when combining this value with others downstream.
    """
    with tf.compat.v1.name_scope(name, 'sqrt_with_finite_grads', [x]):
        x = tf.convert_to_tensor(value=x, name='x')
        if not x.dtype.is_floating:
            raise TypeError('Input `x` must be floating type.')
        def grad(grad_ys):
            large_float_like_x = np.sqrt(np.finfo(x.dtype.as_numpy_dtype()).max)
            safe_grads = tf.where(
                tf.equal(x, 0), tf.fill(tf.shape(input=x), large_float_like_x),
                0.5 * tf.math.rsqrt(x))
            return grad_ys * safe_grads
    return tf.sqrt(x), grad

###
# forward_gradients_v2: Taken from https://github.com/renmengye/tensorflow-forward-ad
###

def forward_gradients_v2(ys, xs, grad_xs=None, gate_gradients=False):
    """Forward-mode pushforward analogous to the pullback defined by tf.gradients.
    With tf.gradients, grad_ys is the vector being pulled back, and here d_xs is
    the vector being pushed forward."""
    if type(ys) == list:
        v = [tf.ones_like(yy) for yy in ys]
    else:
        v = tf.ones_like(ys)  # dummy variable
    g = tf.gradients(ys, xs, grad_ys=v)
    return tf.gradients(g, v, grad_ys=grad_xs)