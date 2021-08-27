import jax.numpy as jnp
from jax import vmap, nn, tree_map, tree_multimap, random
from jax.flatten_util import ravel_pytree
from itertools import product
from jax.lax import scan, while_loop
from jaxns.gaussian_process.kernels import Kernel, StationaryKernel
from jaxns.utils import chunked_pmap
import haiku as hk
from bayes_gain_screens.utils import build_lookup_index, make_coord_array


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

    def compute_integration_limits(self, x, k, bottom, width):
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
            return vmap(lambda x,k: self.compute_integration_limits(x,k,bottom, width))(x,k)
        smin = (bottom - (x[2] - self.x0[2])) / k[2]
        smax = (bottom + width - (x[2] - self.x0[2])) / k[2]
        return smin, smax

    def compute_normal_coordinates(self, x, k, bottom, width):
        """
        Compute the normalised coordinates such that the domain of s1 and s2 is (0, 1).

        Args:
            x: [N, 3]
            k: [N, 3]
            bottom: scalar
            width: scalar
            l: scalar

        Returns:
            x': [N, 3]
            k': [N, 3]
            ds: [N] integration factor

        """
        smin, smax = self.compute_integration_limits(x, k, bottom, width)
        ds = smax - smin
        xp = (x + k*smin[:, None])
        kp = (k * ds[:, None])
        return xp, kp, ds

    def compute_coeffs(self, x1,k1,x2,k2):
        """
        Computes the descriptive coefficients of the distribution given the primed coordinates.

        Args:
            x1: [3]
            k1: [3]
            x2: [3]
            k2: [3]

        Returns:
            (a,b,c,d,e,f) each of which is a scalar

        """
        dx = x1 - x2
        a = dx @ dx  #
        b = 2. * dx @ k1  # s1
        c = -2 * dx @ k2  # s2
        d = k1 @ k1  # s1^2
        e = -2. * k1 @ k2  # s1*s2
        f = k2 @ k2  # s2^2
        return (a,b,c,d,e,f)

    def precompute_kernel_space(self, X, bottom_min=50., bottom_max=500.,width_min=10., width_max=200., l_min=0.5, l_max=40.):
        x = X[:, 0:3]  # N,3
        k = X[:, 3:6]
        def body(state, X):
            (bottom, width, l) = X
            xp,kp,ds = self.compute_normal_coordinates(x, k, bottom, width, l)
            (a,b,c,d,e,f) = vmap(lambda x1,k1: vmap(lambda x2,k2: self.compute_coeffs(x1,k1,x2,k2))(xp,kp))(xp,kp)
            min_vals = tree_map(jnp.min, (a,b,c,d,e,f))
            max_vals = tree_map(jnp.max, (a,b,c,d,e,f))
            return state, (min_vals, max_vals)

        parameter_ranges = list(product((bottom_min, bottom_max), (width_min, width_max), (l_min, l_max)))
        _, (min_vals, max_vals) = scan(body, (), jnp.asarray(parameter_ranges))
        min_vals = tree_map(lambda x: jnp.min(x, axis=0), min_vals)
        max_vals = tree_map(lambda x: jnp.max(x, axis=0), max_vals)
        arrays = tree_multimap(lambda min, max: jnp.linspace(min, max, self.S_grid), min_vals, max_vals)
        lookup_func = build_lookup_index(*arrays)

        arrays = jnp.meshgrid(*arrays, indexing='ij')
        arrays = [a.ravel() for a in arrays]

        return min_vals, max_vals

    def build_Kxy(self, bottom, width, fed_kernel_params):
        """
        Construct a callable that returns the TEC kernel function.

        Args:
            bottom: ionosphere layer bottom in km
            width: ionosphere layer width in km
            fed_kernel_params: dict of FED kernel parameters
                Typically a lengthscale and scaling parameter, but perhaps more.

        Returns:
            callable(x1:[N,3],k1:[N,3],x2:[M,3],k2:[M,3]) -> [N, M]
        """
        sigma = fed_kernel_params.get('sigma')
        l = fed_kernel_params.get('l')

        def ray_integral(f, x, k):
            smin, smax = self.compute_integration_limits(x, k, bottom, width)
            ds = (smax - smin)  # width/(k.z)
            _x = x + k * smin
            _k = k * ds
            t = jnp.linspace(0., 1., self.S_marg + 1)
            return jnp.sum(scan_vmap(lambda t: f(_x + t * _k))(t), axis=0) * (ds / self.S_marg)

        K = lambda x, y: self.fed_kernel(x[None], y[None], l, sigma)[0, 0]
        Ky = lambda x, x2, k2: vmap(lambda x2, k2: ray_integral(lambda y: K(x, y), x2, k2))(x2, k2)
        Kxy = lambda x1, k1, x2, k2: vmap(lambda x1, k1: ray_integral(lambda x: Ky(x, x2, k2), x1, k1))(x1, k1)

        return Kxy

    def __call__(self, X1, X2, bottom, width, fed_kernel_params, wind_velocity=None):
        """
        Computes the Tomographic Kernel.

        Args:
            X1: [N, 6/7]
            X2: [M, 6/7]
            bottom: bottom of ionosphere in km
            width: width of ionosphere in km
            fed_kernel_params: dictionary of FED kernel parameters.
            wind_velocity:

        Returns:

        """
        if not isinstance(fed_kernel_params, dict):
            raise TypeError(f"fed_kernel_params should be a dict, got {type(fed_kernel_params)}")

        x1 = X1[:, 0:3]  # N,3
        k1 = X1[:, 3:6]
        x2 = X2[:, 0:3]  # M,3
        k2 = X2[:, 3:6]

        Kxy = self.build_Kxy(bottom, width, fed_kernel_params)

        def _Kxy(x1,k1,x2,k2):
            x1, k1 = jnp.broadcast_arrays(x1, k1)
            x2, k2 = jnp.broadcast_arrays(x2, k2)
            return Kxy(x1,k1,x2,k2)

        if self.compute_tec:
            if wind_velocity is not None:
                t1 = X1[:, 6:7]
                x1 = x1 - t1 * wind_velocity
                t2 = X2[:, 6:7]
                x2 = x2 - t2 * wind_velocity
            return _Kxy(x1, k1, x2, k2)
        else:
            if wind_velocity is not None:
                t1 = X1[:, 6:7]
                x1 = x1 - t1 * wind_velocity
                t2 = X2[:, 6:7]
                x2 = x2 - t2 * wind_velocity
                return _Kxy(x1, k1, x2, k2) \
                       + _Kxy(self.ref_ant - t1 * wind_velocity, k1, self.ref_ant - t2 * wind_velocity, k2) \
                       - _Kxy(x1, k1, self.ref_ant - t2 * wind_velocity, k2) \
                       - _Kxy(self.ref_ant - t1 * wind_velocity, k1,x2, k2)
            else:
                return _Kxy(x1, k1, x2, k2) \
                       + _Kxy(self.ref_ant, k1, self.ref_ant, k2) \
                       - _Kxy(x1, k1, self.ref_ant, k2) \
                       - _Kxy(self.ref_ant, k1, x2, k2)


class NeuralTomographicKernel(TomographicKernel):
    def __init__(self, x0, ref_ant, fed_kernel: StationaryKernel, S_marg=25, compute_tec=False):
        super(NeuralTomographicKernel, self).__init__(x0, ref_ant, fed_kernel, S_marg=S_marg, compute_tec=compute_tec)


        @hk.without_apply_rng
        @hk.transform
        def neural_model(x1, k1, x2, k2, bottom, width, l, sigma):
            features, ds1, ds2 = self.construct_features(x1, k1, x2, k2, bottom, width, l)
            mlp = hk.Sequential([hk.Linear(32), nn.sigmoid, hk.Linear(16), nn.sigmoid, hk.Linear(1)])
            return sigma**2 * mlp(features) * ds1 * ds2

        self._neural_model = neural_model

    def compute_coeffs(self, x1,k1,x2,k2):
        """
        Computes the descriptive coefficients of the distribution given the normal coordinates.

        Args:
            x1: [3]
            k1: [3]
            x2: [3]
            k2: [3]

        Returns:
            (a,b,c,d,e,f) each of which is a scalar

        """
        dx = x1 - x2
        a = dx @ dx  #
        b = 2. * dx @ k1  # s1
        c = -2 * dx @ k2  # s2
        d = k1 @ k1  # s1^2
        e = -2. * k1 @ k2  # s1*s2
        f = k2 @ k2  # s2^2
        return (a,b,c,d,e,f)

    def construct_features(self, x1, k1, x2, k2, bottom, width, l):
        """
        |x1 + k1*s1 - (x2 + k2 * s2)|^2
        |x1-x2 + (k1*s1 - k2*s2)|^2
        |x1-x2|^2 + (k1*s1 - k2*s2).(k1*s1 - k2*s2) + (k1*s1 - k2*s2).(x1-x2)
        a = |x1-x2|^2
        b = k1.k1*s1^2

        Args:
            x1:
            k1:
            x2:
            k2:
            x0:
            bottom:
            width:
            l:

        Returns:

        """

        x1, k1, ds1 = self.compute_normal_coordinates(x1, k1, bottom, width)
        x2, k2, ds2 = self.compute_normal_coordinates(x2, k2, bottom, width)

        a,b,c,d,e,f = self.compute_coeffs(x1, k1, x2, k2)
        return jnp.asarray([a,b,c,e,d,f])/l, ds1, ds2

    def init_params(self, key):
        kwargs = dict(x1=jnp.zeros(3),
                      k1=jnp.zeros(3),
                      x2=jnp.zeros(3),
                      k2=jnp.zeros(3),
                      bottom=jnp.asarray(0.), width=jnp.asarray(0.),
                      l=jnp.asarray(0.), sigma=jnp.asarray(0.))
        return self._neural_model.init(key,  **kwargs)

    def set_params(self, params):
        self._params = params

    def training_neural_model(self,X, bottom, width, fed_kernel_params, wind_velocity=None):

        true_Kxy = super(NeuralTomographicKernel, self).build_Kxy(bottom, width, fed_kernel_params)

        def get_example(x1, k1, x2, k2):
            self._neural_model()

        init_params = self.init_params(random.PRNGKey(42))
        flat_init_params, unravel_func = ravel_pytree(init_params)

        def loss(flat_params):
            params = unravel_func(flat_params)



    def build_Kxy(self, bottom, width, fed_kernel_params):
        sigma = fed_kernel_params.get('sigma')
        l = fed_kernel_params.get('l')
        def K(x1, k1, x2, k2):
            return self._neural_model.apply(self._params, x1, k1, x2, k2, self.x0, bottom, width, l, sigma)

        def Kxy(x1, k1, x2, k2):
            return vmap(lambda x1, k1:
                        vmap(lambda x2, k2:
                             K(x1, k1, x2, k2))(x2, k2))(x1, k1)
        return Kxy


class TomographicKernelWeighted(TomographicKernel):
    """
    Computes the tomographic kernel using weighted sum of stationary kernel.
    """
    def __init__(self, x0, ref_ant, fed_kernel: StationaryKernel, S_marg=25, compute_tec=False):
        super(TomographicKernelWeighted, self).__init__(x0, ref_ant, fed_kernel, S_marg=S_marg, compute_tec=compute_tec)

    def gamma_squared(self, s1, s2, a, b, c, d, e, f):
        return a + (b + d * s1) * s1 + (c + f * s2 + e * s1) * s2

    def compute_coeffs(self, x1,k1,x2,k2):
        """
        Computes the descriptive coefficients of the distribution given the normal coordinates.

        Args:
            x1: [3]
            k1: [3]
            x2: [3]
            k2: [3]

        Returns:
            (a,b,c,d,e,f) each of which is a scalar

        """
        dx = x1 - x2
        a = dx @ dx  #
        b = 2. * dx @ k1  # s1
        c = -2 * dx @ k2  # s2
        d = k1 @ k1  # s1^2
        e = -2. * k1 @ k2  # s1*s2
        f = k2 @ k2  # s2^2
        return (a,b,c,d,e,f)

    def build_Kxy(self, bottom, width, fed_kernel_params):
        # gamma_squared is dimensionless distance parametrisation

        sigma = fed_kernel_params.get('sigma')
        l = fed_kernel_params.get('l')

        def compute_pdf(x1,k1,x2,k2):
            s1 = jnp.linspace(0., 1., self.S_marg + 1)
            s2 = jnp.linspace(0., 1., self.S_marg + 1)

            def gamma(s1, s2):
                return jnp.linalg.norm((x1-x2)+s1*k1 - s2*k2)/l

            pdf, bins = jnp.histogram(vmap(lambda s1:
                                        vmap(lambda s2:
                                             gamma(s1,s2))(s2))(s1),
                                bins=self.S_marg,
                                   density=True)
            return pdf, bins

        def compute_pdf_from_coeffs(a,b,c,d,e,f):
            s1 = jnp.linspace(0., 1., self.S_marg + 1)
            s2 = jnp.linspace(0., 1., self.S_marg + 1)

            def gamma(s1, s2):
                return jnp.sqrt(jnp.maximum(0., self.gamma_squared(s1, s2, a, b, c, d, e, f)))/l
            pdf, bins = jnp.histogram(vmap(lambda s1:
                                        vmap(lambda s2:
                                             gamma(s1,s2))(s2))(s1),
                                bins=self.S_marg,
                                   density=True)
            return pdf, bins



        def K(x1,k1,x2,k2, ds1,ds2):
            # compute pdf from a,b,c,d,e,f
            pmf, bins = compute_pdf(x1,k1,x2,k2)


            # a,b,c,d,e,f = self.compute_coeffs(x1, k1, x2, k2)
            # pmf, bins = compute_pdf_from_coeffs(a,b,c,d,e,f)

            dgamma = bins[1] - bins[0]
            dpdf = pmf * dgamma

            K_integrand = vmap(lambda gamma: jnp.exp(self.fed_kernel.act(gamma ** 2, 1.)))(bins)
            K_integrand = 0.5 * (K_integrand[:-1] + K_integrand[1:])

            return sigma**2 * jnp.sum(dpdf*K_integrand) * ds1 * ds2

        def Kxy(x1, k1, x2, k2):
            x1, k1, ds1 = self.compute_normal_coordinates(x1, k1, bottom, width)
            x2, k2, ds2 = self.compute_normal_coordinates(x2, k2, bottom, width)

            return vmap(lambda x1,k1, ds1:
                        vmap(lambda x2, k2, ds2:
                             K(x1,k1,x2,k2,ds1,ds2))(x2,k2,ds2))(x1,k1,ds1)


        return Kxy