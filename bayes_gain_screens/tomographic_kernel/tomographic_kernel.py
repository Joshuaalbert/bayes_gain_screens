import jax.numpy as jnp
from jax import vmap, tree_map, tree_multimap
from itertools import product
from jax.lax import scan
from jaxns.gaussian_process.kernels import Kernel, StationaryKernel
from typing import NamedTuple

from bayes_gain_screens.utils import build_lookup_index


def scan_vmap(f):
    """
    Applies a function `f` over a pytree, over the first dimension of each leaf.
    Similar to vmap, but sequentially executes, rather that broadcasting.
    """
    def run(*args):
        def body(state, X):
            return state, f(*X)

        _, results = scan(body, (), args)
        return results

    return run


class GeodesicTuple(NamedTuple):
    x:jnp.ndarray
    k:jnp.ndarray
    t:jnp.ndarray
    ref_x:jnp.ndarray

class TomographicKernel(Kernel):
    def __init__(self, x0, earth_centre, fed_kernel: StationaryKernel, S_marg=25, compute_tec=False):
        """
        Tomographic model with curved thick layer above the Earth.

        Args:
            x0: [3] jnp.ndarray The array center. We assume a spherical Earth and that height is referenced radially from this point.
            earth_centre: Earth's centre
            fed_kernel: StationaryKernel, covariance function of the free-electron density.
            S_marg: int, the resolution of quadrature
            compute_tec: bool, whether to compute TEC or DTEC.
        """
        self.S_marg = S_marg
        self.x0 = x0 - earth_centre
        self.earth_centre = earth_centre
        self.fed_kernel = fed_kernel
        self.compute_tec = compute_tec

    def compute_integration_limits_flat(self, x, k, bottom, width):
        """
        Compute the integration limits of the flat layer ionosphere.

        Args:
            x: [3] or [N, 3]
            k: [3] or [N, 3]
            bottom: scalar height of bottom of layer
            width: scalar width of width of layer

        Returns:
            s_min, s_max with shapes:
                - scalars if x and k are [3]
                - arrays of [N] if x or k is [N,3]
        """
        if (len(x.shape) == 2) or (len(k.shape) == 2):
            x,k = jnp.broadcast_arrays(x, k)
            return vmap(lambda x,k: self.compute_integration_limits_flat(x,k,bottom, width))(x,k)
        smin = (bottom - (x[2] - self.x0[2])) / k[2]
        smax = (bottom + width - (x[2] - self.x0[2])) / k[2]
        return smin, smax

    def compute_integration_limits(self, x, k, bottom, width):
        """
        Compute the integration limits of the curved layer ionosphere.

        Args:
            x: [3] or [N, 3]
            k: [3] or [N, 3]
            bottom: scalar height of bottom of layer
            width: scalar width of width of layer

        Returns:
            s_min, s_max with shapes:
                - scalars if x and k are [3]
                - arrays of [N] if x or k is [N,3]
        """
        if (len(x.shape) == 2) or (len(k.shape) == 2):
            x,k = jnp.broadcast_arrays(x, k)
            return vmap(lambda x,k: self.compute_integration_limits(x,k,bottom, width))(x,k)
        x0_hat = self.x0 / jnp.linalg.norm(self.x0)
        bottom_radius2 = jnp.sum(jnp.square(self.x0 + bottom * x0_hat))
        top_radius2 = jnp.sum(jnp.square(self.x0 + (bottom + width) * x0_hat))
        xk = x @ k
        x2 = x @ x
        smin = -xk + jnp.sqrt(xk**2 + (bottom_radius2 - x2))
        smax = -xk + jnp.sqrt(xk**2 + (top_radius2 - x2))
        return smin, smax

    def frozen_flow_transform(self, t, y, bottom, wind_velocity=None):
        """
        Computes the frozen flow transform on the coordinates.

        Args:
            t: time in seconds
            y: position in km, origin at centre of Earth
            bottom: bottom of ionosphere in km
            wind_velocity: layer velocity in km

        Returns:
            Coordinates inverse rotating the coordinates to take into account the flow of ionosphere around
            surface of Earth.
        """
        # rotate around Earth's core
        if t is None:
            return y
        if wind_velocity is None:
            return y

        rotation_axis = jnp.cross(wind_velocity, self.x0)
        rotation_axis /= jnp.linalg.norm(rotation_axis)
        # v(r) = theta_dot * r
        # km/s / km
        theta_dot = jnp.linalg.norm(wind_velocity) / (bottom + jnp.linalg.norm(self.x0))

        angle = - theta_dot * t
        # Rotation
        u_cross_x = jnp.cross(rotation_axis, y)
        rotated_y = rotation_axis * (rotation_axis @ y) \
                  + jnp.cos(angle) * jnp.cross(u_cross_x, rotation_axis) \
                  + jnp.sin(angle) * u_cross_x
        # print(rotated_y - y, t*wind_velocity, wind_velocity)
        return rotated_y

    def build_Kxy(self, bottom, width, fed_sigma, fed_kernel_params, wind_velocity=None):
        """
        Construct a callable that returns the TEC kernel function.

        Args:
            bottom: ionosphere layer bottom in km
            width: ionosphere layer width in km
            fed_sigma: variation scaling in mTECU/km, or 10^10 electron/m^3
            fed_kernel_params: dict of FED kernel parameters
                Typically a lengthscale and scaling parameter, but perhaps more.

        Returns:
            callable(x1:[N,3],k1:[N,3],x2:[M,3],k2:[M,3]) -> [N, M]
        """
        sigma = fed_kernel_params.get('sigma')
        l = fed_kernel_params.get('l')

        def ray_integral(f):
            t = jnp.linspace(0., 1., self.S_marg + 1)
            return jnp.sum(vmap(f)(t), axis=0) * (1. / self.S_marg)
            
        def build_geodesic(x, k, t):
            smin, smax = self.compute_integration_limits(x, k, bottom, width)
            def g(epsilon):
                y = x + k * (smin + (smax - smin) * epsilon)
                return self.frozen_flow_transform(t, y, bottom=bottom, wind_velocity=wind_velocity)
            return g, (smax - smin)
        
        def integrate_integrand(X1:GeodesicTuple, X2:GeodesicTuple):
            g1, ds1 = build_geodesic(X1.x, X1.k, X1.t)
            g1_ref, ds1_ref = build_geodesic(X1.ref_x, X1.k, X1.t)
            g2, ds2 = build_geodesic(X2.x, X2.k, X2.t)
            g2_ref, ds2_ref = build_geodesic(X2.ref_x, X2.k, X2.t)
            def f(epsilon_1, epsilon_2):
                results = (ds1 * ds2) * self.fed_kernel(g1(epsilon_1)[None], g2(epsilon_2)[None], l, sigma)
                if not self.compute_tec:
                    results += (ds1_ref * ds2_ref) * self.fed_kernel(g1_ref(epsilon_1)[None], g2_ref(epsilon_2)[None], l, sigma)
                    results -= (ds1 * ds2_ref) * self.fed_kernel(g1(epsilon_1)[None], g2_ref(epsilon_2)[None], l, sigma)
                    results -= (ds1_ref * ds2) * self.fed_kernel(g1_ref(epsilon_1)[None], g2(epsilon_2)[None], l, sigma)
                return results[0,0]
            return ray_integral(lambda epsilon_2: ray_integral(lambda epsilon_1: f(epsilon_1, epsilon_2)))

        def Kxy(X1:GeodesicTuple, X2:GeodesicTuple):
            return fed_sigma**2 * scan_vmap(lambda X1: vmap(lambda X2: integrate_integrand(X1, X2))(X2))(X1)

        return Kxy

    def build_mean_func(self, bottom, width, fed_mu, wind_velocity=None):
        """
        Construct a callable that returns the TEC kernel function.

        Args:
            bottom: ionosphere layer bottom in km
            width: ionosphere layer width in km
            fed_sigma: variation scaling in mTECU/km, or 10^10 electron/m^3
            fed_kernel_params: dict of FED kernel parameters
                Typically a lengthscale and scaling parameter, but perhaps more.

        Returns:
            callable(x1:[N,3],k1:[N,3],x2:[M,3],k2:[M,3]) -> [N, M]
        """

        def geodesic_intersection(x, k, t):
            smin, smax = self.compute_integration_limits(x, k, bottom, width)
            return (smax - smin)

        def layer_intersection(X1: GeodesicTuple):
            ds1 = geodesic_intersection(X1.x, X1.k, X1.t)
            if self.compute_tec:
                return ds1
            ds1_ref = geodesic_intersection(X1.ref_x, X1.k, X1.t)
            return ds1 - ds1_ref

        def mean_func(X1: GeodesicTuple):
            return fed_mu * vmap(layer_intersection)(X1)

        return mean_func

    def mean_function(self, X1: GeodesicTuple, bottom, width, fed_mu, wind_velocity=None):

        mean_func = self.build_mean_func(bottom, width, fed_mu, wind_velocity=wind_velocity)

        def _mean_func(X1:GeodesicTuple):
            X1 = X1._replace(x = X1.x - self.earth_centre, ref_x=X1.ref_x - self.earth_centre)
            X1 = GeodesicTuple(*jnp.broadcast_arrays(*X1))
            return mean_func(X1)

        return _mean_func(X1)


    def __call__(self, X1:GeodesicTuple, X2:GeodesicTuple, bottom, width, fed_sigma, fed_kernel_params, wind_velocity=None):
        """
        Computes the Tomographic Kernel.

        Args:
            X1: GeodesicTuple
            X2: GeodesicTuple or None
            bottom: bottom of ionosphere in km
            width: width of ionosphere in km
            fed_mu: variation scaling in mTECU/km, or 10^10 electron/m^3
            wind_velocity: in East-North-Up frame

        Returns:

        """
        if not isinstance(fed_kernel_params, dict):
            raise TypeError(f"fed_kernel_params should be a dict, got {type(fed_kernel_params)}")

        Kxy = self.build_Kxy(bottom, width, fed_sigma, fed_kernel_params, wind_velocity=wind_velocity)


        def _Kxy(X1:GeodesicTuple, X2:GeodesicTuple):
            X1 = X1._replace(x = X1.x - self.earth_centre, ref_x=X1.ref_x - self.earth_centre)
            X2 = X2._replace(x = X2.x - self.earth_centre, ref_x=X2.ref_x - self.earth_centre)
            X1 = GeodesicTuple(*jnp.broadcast_arrays(*X1))
            X2 = GeodesicTuple(*jnp.broadcast_arrays(*X2))
            return Kxy(X1, X2)

        return _Kxy(X1, X2)
