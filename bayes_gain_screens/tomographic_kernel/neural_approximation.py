import haiku as hk
from jax import random, numpy as jnp, nn, vmap, value_and_grad, tree_multimap, tree_map, jit
from jax.lax import scan
from bayes_gain_screens.tomographic_kernel.tomographic_kernel import TomographicKernel, NeuralTomographicKernel
from jaxns.gaussian_process.kernels import RBF
from h5parm.utils import create_empty_datapack
from h5parm import DataPack
import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
from bayes_gain_screens.frames import ENU
from bayes_gain_screens.utils import make_coord_array
import pylab as plt


def train_neural_network(datapack: DataPack, batch_size, learning_rate, num_batches):

    with datapack:
        select = dict(pol=slice(0, 1, 1), ant=None, time=slice(0,1,1))
        datapack.current_solset = 'sol000'
        datapack.select(**select)
        axes = datapack.axes_tec
        patch_names, directions = datapack.get_directions(axes['dir'])
        antenna_labels, antennas = datapack.get_antennas(axes['ant'])
        timestamps, times = datapack.get_times(axes['time'])

    antennas = ac.ITRS(*antennas.cartesian.xyz, obstime=times[0])
    ref_ant = antennas[0]
    frame = ENU(obstime=times[0], location=ref_ant.earth_location)
    antennas = antennas.transform_to(frame)
    ref_ant = antennas[0]
    directions = directions.transform_to(frame)
    x = antennas.cartesian.xyz.to(au.km).value.T
    k = directions.cartesian.xyz.value.T
    t = times.mjd
    t -= t[len(t)//2]
    t *= 86400.
    n_screen = 250
    kstar = random.uniform(random.PRNGKey(29428942),(n_screen,3), minval=jnp.min(k, axis=0), maxval=jnp.max(k, axis=0))
    kstar /= jnp.linalg.norm(kstar, axis=-1, keepdims=True)
    X = jnp.asarray(make_coord_array(x,
                                     jnp.concatenate([k,kstar], axis=0),
                                     t[:,None]))
    x0 = jnp.asarray(antennas.cartesian.xyz.to(au.km).value.T[0, :])
    ref_ant = x0

    kernel = TomographicKernel(x0, ref_ant, RBF(), S_marg=100)
    neural_kernel = NeuralTomographicKernel(x0, ref_ant)

    def loss(params, key):
        keys = random.split(key,5)
        indices = random.permutation(keys[0], jnp.arange(X.shape[0]))[:batch_size]
        X_batch = X[indices, :]

        wind_velocity = random.uniform(keys[1], shape=(3,), minval=jnp.asarray([-200., -200., 0.]), maxval=jnp.asarray([200., 200., 0.]))/1000.
        bottom = random.uniform(keys[2], minval=50., maxval=500.)
        width = random.uniform(keys[3], minval=40., maxval=300.)
        l = random.uniform(keys[4], minval=1., maxval=30.)
        sigma = 1.
        K = kernel(X_batch, X_batch, bottom, width, l, sigma, wind_velocity=wind_velocity)
        neural_kernel.set_params(params)
        neural_K = neural_kernel(X_batch, X_batch, bottom, width, l, sigma, wind_velocity=wind_velocity)

        return jnp.mean((K-neural_K)**2)/width**2

    init_params = neural_kernel.init_params(random.PRNGKey(42))

    def train_one_batch(params, key):
        l, g = value_and_grad(lambda params: loss(params, key))(params)
        params = tree_multimap(lambda p, g: p - learning_rate*g, params, g)
        return params, l

    final_params, losses = jit(lambda key: scan(train_one_batch, init_params, random.split(key, num_batches)))(random.PRNGKey(42))

    plt.plot(losses)
    plt.yscale('log')
    plt.show()


if __name__ == '__main__':
    datapack = create_empty_datapack(250, 2, 100, pols=None,
                          field_of_view_diameter=8.,
                          start_time=None,
                          time_resolution=30.,
                          min_freq=122.,
                          max_freq=166.,
                          array_file=None,
                          phase_tracking=None,
                          save_name='test_datapack.h5',
                          clobber=True)
    train_neural_network(datapack, batch_size=64, learning_rate=0.001, num_batches=100)