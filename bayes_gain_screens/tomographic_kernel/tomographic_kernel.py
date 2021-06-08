import jax.numpy as jnp
from jax import vmap
from jax.lax import scan
from jaxns.gaussian_process.kernels import Kernel, StationaryKernel


def scan_vmap(f):
    def run(*args):
        def body(state, X):
            return state, f(*X)

        _, results = scan(body, (), args)
        return results

    return run


def tec_kernel(x1, k1, x2, k2, x0, bottom, width, l, sigma, S_marg, fed_kernel, *fed_kernel_params):
    def ray_integral(f, x, k):
        smin = (bottom - (x[2] - x0[2])) / k[2]
        smax = (bottom + width - (x[2] - x0[2])) / k[2]
        ds = (smax - smin)
        _x = x + k * smin
        _k = k * ds
        t = jnp.linspace(0., 1., S_marg + 1)
        return jnp.sum(scan_vmap(lambda t: f(_x + t * _k))(t), axis=0) * (ds / S_marg)

    K = lambda x, y: fed_kernel(x[None], y[None], l, sigma, *fed_kernel_params)[
        0, 0]  # * self.fed_kernel(x[None]-h, y[None]-h, width, 1., *fed_kernel_params)[0,0]
    Ky = lambda x, x2, k2: vmap(lambda x2, k2: ray_integral(lambda y: K(x, y), x2, k2))(*jnp.broadcast_arrays(x2, k2))
    Kxy = lambda x1, k1, x2, k2: vmap(lambda x1, k1: ray_integral(lambda x: Ky(x, x2, k2), x1, k1))(
        *jnp.broadcast_arrays(x1, k1))
    return Kxy(x1, k1, x2, k2)


class TomographicKernel(Kernel):
    def __init__(self, x0, ref_ant, fed_kernel: StationaryKernel, S_marg=25, compute_tec=False):
        self.S_marg = S_marg
        self.x0 = x0
        self.ref_ant = ref_ant
        self.fed_kernel = fed_kernel
        self.compute_tec = compute_tec

    def __call__(self, X1, X2, bottom, width, l, sigma, *fed_kernel_params, wind_velocity=None):
        x1 = X1[:, 0:3]  # N,3
        k1 = X1[:, 3:6]
        x2 = X2[:, 0:3]  # M,3
        k2 = X2[:, 3:6]

        def ray_integral(f, x, k):
            smin = (bottom - (x[2] - self.x0[2])) / k[2]
            smax = (bottom + width - (x[2] - self.x0[2])) / k[2]
            ds = (smax - smin)
            _x = x + k * smin
            _k = k * ds
            t = jnp.linspace(0., 1., self.S_marg + 1)
            return jnp.sum(scan_vmap(lambda t: f(_x + t * _k))(t), axis=0) * (ds / self.S_marg)

        K = lambda x, y: self.fed_kernel(x[None], y[None], l, sigma, *fed_kernel_params)[0, 0]
        Ky = lambda x, x2, k2: vmap(lambda x2, k2: ray_integral(lambda y: K(x, y), x2, k2))(
            *jnp.broadcast_arrays(x2, k2))
        Kxy = lambda x1, k1, x2, k2: vmap(lambda x1, k1: ray_integral(lambda x: Ky(x, x2, k2), x1, k1))(
            *jnp.broadcast_arrays(x1, k1))

        if self.compute_tec:
            if wind_velocity is not None:
                t1 = X1[:, 6:7]
                x1 = x1 - t1 * wind_velocity
                t2 = X2[:, 6:7]
                x2 = x2 - t2 * wind_velocity
                return Kxy(x1, k1, x2, k2)
        else:
            if wind_velocity is not None:
                t1 = X1[:, 6:7]
                x1 = x1 - t1 * wind_velocity
                t2 = X2[:, 6:7]
                x2 = x2 - t2 * wind_velocity
                return Kxy(x1, k1, x2, k2) + Kxy(self.ref_ant - t1 * wind_velocity, k1,
                                                 self.ref_ant - t2 * wind_velocity, k2) \
                       - Kxy(x1, k1, self.ref_ant - t2 * wind_velocity, k2) - Kxy(self.ref_ant - t1 * wind_velocity, k1,
                                                                                  x2, k2)
