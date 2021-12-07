from jax.config import config
config.update("jax_enable_x64", True)

from bayes_gain_screens.tomographic_kernel import TomographicKernel, GeodesicTuple
from bayes_gain_screens.tomographic_kernel.debug import debug_inference
from bayes_gain_screens.utils import make_coord_array
from bayes_gain_screens.plotting import plot_vornoi_map
from bayes_gain_screens.frames import ENU
from h5parm import DataPack
from jaxns.gaussian_process.kernels import RBF, M12
import jax.numpy as jnp
from jax import jit, random, vmap, nn
from h5parm.utils import make_example_datapack
import astropy.units as au
import astropy.coordinates as ac
import pylab as plt
import haiku as hk


def test_compare_with_forward_model():
    dp = make_example_datapack(5, 24, 1, clobber=True)
    with dp:
        select = dict(pol=slice(0, 1, 1), ant=slice(0, None, 1))
        dp.current_solset = 'sol000'
        dp.select(**select)
        tec_mean, axes = dp.tec
        tec_mean = tec_mean[0, ...]
        patch_names, directions = dp.get_directions(axes['dir'])
        antenna_labels, antennas = dp.get_antennas(axes['ant'])
        timestamps, times = dp.get_times(axes['time'])

    antennas = ac.ITRS(*antennas.cartesian.xyz, obstime=times[0])
    ref_ant = antennas[0]
    earth_centre = ac.ITRS(x=0 * au.m, y=0 * au.m, z=0. * au.m, obstime=times[0])
    frame = ENU(obstime=times[0], location=ref_ant.earth_location)
    antennas = antennas.transform_to(frame)
    earth_centre = earth_centre.transform_to(frame)
    ref_ant = antennas[0]
    directions = directions.transform_to(frame)
    x = antennas.cartesian.xyz.to(au.km).value.T[20:21, :]
    k = directions.cartesian.xyz.value.T
    X = make_coord_array(x, k)
    x0 = ref_ant.cartesian.xyz.to(au.km).value
    earth_centre = earth_centre.cartesian.xyz.to(au.km).value
    bottom = 200.
    width = 50.
    l = 10.
    sigma = 1.
    fed_kernel_params = dict(l=l, sigma=sigma)
    S_marg = 1000
    fed_kernel = M12()

    def get_points_on_rays(X):
        x = X[0:3]
        k = X[3:6]
        smin = (bottom - (x[2] - x0[2])) / k[2]
        smax = (bottom + width - (x[2] - x0[2])) / k[2]
        ds = (smax - smin)
        _x = x + k * smin
        _k = k * ds
        t = jnp.linspace(0., 1., S_marg + 1)
        return _x + _k * t[:, None], ds/S_marg

    points, ds = vmap(get_points_on_rays)(X)
    points = points.reshape((-1, 3))
    plt.scatter(points[:,1], points[:,2], marker='.')
    plt.show()


    K = fed_kernel(points, points,l, sigma)
    plt.imshow(K)
    plt.show()
    L = jnp.linalg.cholesky(K + 1e-6*jnp.eye(K.shape[0]))
    Z = L@random.normal(random.PRNGKey(1245214),(L.shape[0],3000))
    Z = Z.reshape((directions.shape[0],-1, Z.shape[1]))
    Y = jnp.sum(Z * ds[:, None, None], axis=1)
    K = jnp.mean(Y[:, None, :]*Y[None, :, :], axis=2)
    # print("Directly Computed TEC Covariance",K)
    plt.imshow(K)
    plt.colorbar()
    plt.title("Directly Computed TEC Covariance")
    plt.show()

    # kernel = TomographicKernel(x0, x0,fed_kernel, S_marg=200, compute_tec=False)
    # K = kernel(X, X, bottom, width, fed_kernel_params)
    # plt.imshow(K)
    # plt.colorbar()
    # plt.title("Analytic Weighted TEC Covariance")
    # plt.show()
    #
    # print("Analytic Weighted TEC Covariance",K)
    # print(x0, earth_centre, fed_kernel)
    kernel = TomographicKernel(x0, earth_centre, fed_kernel, S_marg=200, compute_tec=True)
    # print(X)
    X1 = GeodesicTuple(x = X[:,0:3], k = X[:,3:6], t=jnp.zeros_like(X[:,:1]),ref_x=x0)
    print(X1)
    K = kernel(X1, X1, bottom, width, fed_kernel_params)
    plt.imshow(K)
    plt.colorbar()
    plt.title("Analytic TEC Covariance")
    plt.show()
    print("Analytic TEC Covariance",K)

def test_train_neural_network():
    def model(x, is_training=False):
        #x1, k1, x2, k2, x0, bottom, width, lengthscale -> 16
        mlp = hk.Sequential([hk.Linear(32), nn.relu, hk.Linear(16), nn.relu, hk.Linear(1)])
        return mlp(x)

    model = hk.without_apply_rng(hk.transform(model))
    init_params = model.init(random.PRNGKey(2345), jnp.zeros(16))
    print(model.apply())


def test_tomographic_kernel():
    dp = make_example_datapack(500,24, 1, clobber=True)
    with dp:
        select = dict(pol=slice(0, 1, 1), ant=slice(0, None, 1))
        dp.current_solset = 'sol000'
        dp.select(**select)
        tec_mean, axes = dp.tec
        tec_mean = tec_mean[0, ...]
        patch_names, directions = dp.get_directions(axes['dir'])
        antenna_labels, antennas = dp.get_antennas(axes['ant'])
        timestamps, times = dp.get_times(axes['time'])
    antennas = ac.ITRS(*antennas.cartesian.xyz, obstime=times[0])
    ref_ant = antennas[0]
    frame = ENU(obstime=times[0], location=ref_ant.earth_location)
    antennas = antennas.transform_to(frame)
    ref_ant = antennas[0]
    directions = directions.transform_to(frame)
    x = antennas.cartesian.xyz.to(au.km).value.T
    k = directions.cartesian.xyz.value.T
    X = make_coord_array(x[50:51,:], k)
    x0 = ref_ant.cartesian.xyz.to(au.km).value
    print(k.shape)

    kernel = TomographicKernel(x0, x0, RBF(),S_marg=25)
    K = jit(lambda X: kernel(X, X, bottom=200., width=50., fed_kernel_params=dict(l=7., sigma=1.)))(jnp.asarray(X))
    # K /= jnp.outer(jnp.sqrt(jnp.diag(K)), jnp.sqrt(jnp.diag(K)))
    plt.imshow(K)
    plt.colorbar()
    plt.show()
    L = jnp.linalg.cholesky(K+1e-6*jnp.eye(K.shape[0]))
    print(L)
    dtec = L @ random.normal(random.PRNGKey(24532), shape=(K.shape[0],))
    print(jnp.std(dtec))
    ax = plot_vornoi_map(k[:,0:2],dtec)
    ax.set_xlabel(r"$k_{\rm east}$")
    ax.set_ylabel(r"$k_{\rm north}$")
    ax.set_xlim(-0.1, 0.1)
    ax.set_ylim(-0.1, 0.1)
    plt.show()


def test_plot_data():
    dp = DataPack('/home/albert/data/gains_screen/data/L342938_DDS5_full_merged.h5', readonly=True)
    with dp:
        select = dict(pol=slice(0, 1, 1), ant=[0, 50], time=slice(0,100,1))
        dp.current_solset = 'sol000'
        dp.select(**select)
        tec_mean, axes = dp.tec
        tec_mean = tec_mean[0, ...]
        const_mean, axes = dp.const
        const_mean = const_mean[0, ...]
        tec_std, axes = dp.weights_tec
        tec_std = tec_std[0, ...]
        patch_names, directions = dp.get_directions(axes['dir'])
        antenna_labels, antennas = dp.get_antennas(axes['ant'])
        timestamps, times = dp.get_times(axes['time'])
    antennas = ac.ITRS(*antennas.cartesian.xyz, obstime=times[0])
    ref_ant = antennas[0]

    for i, time in enumerate(times):
        frame = ENU(obstime=time, location=ref_ant.earth_location)
        antennas = antennas.transform_to(frame)
        directions = directions.transform_to(frame)
        x = antennas.cartesian.xyz.to(au.km).value.T
        k = directions.cartesian.xyz.value.T


        dtec = tec_mean[:,1,i]
        ax = plot_vornoi_map(k[:,0:2],dtec)
        ax.set_title(time)
        plt.show()


if __name__ == '__main__':
    debug_inference()