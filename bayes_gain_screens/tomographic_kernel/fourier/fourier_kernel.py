import numpy as np
from jax import numpy as jnp, vmap
from jaxns.gaussian_process.kernels import StationaryKernel

class FourierKernel(StationaryKernel):

    def __init__(self, spectrum, r_inner, r_outer):
        self._spectrum = spectrum
        N = (int(2*r_outer/r_inner)//2)*2
        r_space = jnp.linspace(r_inner, r_outer, N)
        (k_space,) = fft_freqs(r_space)
        S = vmap(spectrum)(k_space)
        S = jnp.where(k_space == 0, 0., S)
        f = inv_fourier(S, r_space)


        import pylab as plt
        print(S)
        plt.plot(r_space,f)
        plt.show()
        plt.plot(k_space, S)
        plt.ylim(0., 10)
        plt.show()


    def act(self, r2, sigma):
        return 2. * jnp.log(sigma) - 0.5 * r2

def fourier(a, *coords):
    """
    Evaluates

    F[a](s) = int a(x) e^{-2pi i s x} dx
    A(k ds) = sum_m a(x_m) e^{-2pi i k ds (x0 + corner_indices dx)} dx
            = e^{-2pi i k ds x0} dx sum_m a(x_m) e^{-2pi i k ds corner_indices dx}
    dx ds = 1/n => ds = 1/(dx n)
    ds x0 = k ds x0 = k/n * x0/dx
    """

    factor = fft_factor(*coords)
    # return jnp.fft.fftshift(np.fft.fftn(a, a.shape) * factor)
    return jnp.fft.fftshift(np.fft.fftn(a) * factor)


def inv_fourier(A, *coords):
    factor = ifft_factor(*coords)
    # return jnp.fft.ifftn(np.fft.ifftshift(A) * factor, A.shape)
    return  jnp.fft.ifftn(np.fft.ifftshift(A) * factor)


def fft_freqs(*coords):
    s = []
    for i, c in enumerate(coords):
        dx = c[1] - c[0]
        s.append(np.fft.fftshift(np.fft.fftfreq(c.size, dx)))
    return tuple(s)


def ifft_freqs(*coords):
    s = []
    for i, c in enumerate(coords):
        dx = c[1] - c[0]
        s.append(np.fft.fftshift(np.fft.fftfreq(c.size, dx)))
    return tuple(s)


def test_coords_transformations():
    import astropy.units as au
    x = jnp.linspace(-10,10,101)*au.km
    (s,) = fft_freqs(x)
    _x = ifft_freqs(s)
    assert jnp.isclose(_x , x).all()


def fft_factor(*coords):
    def _add_dims(t, i):
        dims = list(range(len(coords)))
        del dims[i]
        return jnp.expand_dims(t, tuple(dims))

    log_factors = 0.
    dx_factor = 1.
    for i, c in enumerate(coords):
        dx = c[1] - c[0]
        x0 = c[0]
        s = jnp.fft.fftfreq(c.size, dx)
        log_factor =  - 2j * jnp.pi * s * x0
        dx_factor *= dx
        log_factors = log_factors + _add_dims(log_factor, i)
    factor = jnp.exp(log_factors)*dx_factor
    return factor


def ifft_factor(*coords):
    def _add_dims(t, i):
        dims = list(range(len(coords)))
        del dims[i]
        return jnp.expand_dims(t, tuple(dims))

    log_factors = 0.
    dx_factor = 1.
    for i, c in enumerate(coords):
        dx = c[1] - c[0]
        x0 = c[0]
        s = jnp.fft.fftfreq(c.size, dx)
        dx_factor /= dx
        log_factor = 2j * jnp.pi * s * x0
        log_factors = log_factors + _add_dims(log_factor, i)
    factor = jnp.exp(log_factors)*dx_factor
    return factor


def test_fourier():
    import astropy.units as au
    def f(x):
        return jnp.exp(-np.pi * x.value ** 2)

    import pylab as plt
    x = jnp.linspace(-10., 10., 101)*au.km
    a = f(x)

    F = fourier(a, x)
    (s,) = fft_freqs(x)
    _a = inv_fourier(F, x)

    plt.plot(s, f(s), label='A true')
    plt.plot(s, jnp.real(F), label='A numeric')
    plt.legend()
    plt.show()

    plt.plot(x, a, label='a')
    plt.plot(x, _a, label='a rec')
    plt.legend()
    # plt.ylim(-10,3)
    plt.show()

    fourier_kernel = FourierKernel(lambda k: jnp.abs(k)**(-11./3.), 0.1, 100)
