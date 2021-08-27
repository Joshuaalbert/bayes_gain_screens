import os
import time

import numpy as np
import astropy.coordinates as ac
import astropy.units as au
from astropy import wcs
from astropy.io import fits

from jax import numpy as jnp, vmap, grad
from jax.scipy.ndimage import map_coordinates

from jax.scipy.signal import convolve

from bayes_gain_screens.frames import ENU

from h5parm import DataPack

import sympy

import logging

logger = logging.getLogger(__name__)
dist_type = au.km
angle_type = au.rad

def build_lookup_index(*arrays):
    def linear_lookup(values, *coords):
        fractional_coordinates = jnp.asarray([jnp.interp(coord, array, jnp.arange(array.size))
                                              for array, coord in zip(arrays, coords)])
        return map_coordinates(values, fractional_coordinates, order=1)
    return linear_lookup

def voronoi_finite_polygons_2d(vor, radius):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    # if radius is None:
    #     radius = np.max(np.linalg.norm(points - np.mean(points, axis=0),axis=1))
    #     # radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def windowed_nanmean(a, w, mode='reflect', axis=0):
    if w is None:
        return jnp.broadcast_to(jnp.nanmean(a, axis=axis, keepdims=True), a.shape)
    count = windowed_sum(jnp.where(jnp.isnan(a), 0., 1.), w, mode, axis)
    return jnp.where(count == 0., 0., windowed_sum(jnp.where(jnp.isnan(a), 0., a), w, mode, axis) / count)


def test_windowed_nanmean():
    a = jnp.array([0., 1., 2., jnp.nan])
    assert jnp.allclose(windowed_nanmean(a, 3), jnp.array([0.6666667, 1., 1.5, 2.]))

def windowed_sum(a, w, mode='reflect', axis=0):
    """
    Perform rolling window average (smoothing).

    Args:
        a: array
        w: window length
        mode: see jnp.pad
        axis: axis to smooth down

    Returns: Array the same size as `a`
    """
    if axis != 0:
        a = jnp.moveaxis(a, axis, 0)
    dims = len(a.shape)
    a = a
    kernel = jnp.reshape(jnp.ones(w), [w] + [1] * (dims - 1))
    _w1 = (w - 1) // 2
    _w2 = _w1 if (w % 2 == 1) else _w1 + 1
    pad_width = [(_w1, _w2)] + [(0, 0)] * (dims - 1)
    a = jnp.pad(a, pad_width=pad_width, mode=mode)
    result = convolve(a, kernel, mode='valid', precision=None)
    if axis != 0:
        result = jnp.moveaxis(result, 0, axis)
    return result

def windowed_mean(a, w, mode='reflect', axis=0):
    """
    Perform rolling window average (smoothing).

    Args:
        a: array
        w: window length
        mode: see jnp.pad
        axis: axis to smooth down

    Returns: Array the same size as `a`
    """
    if w is None:
        return jnp.broadcast_to(jnp.mean(a, axis=axis, keepdims=True), a.shape)
    return windowed_sum(a, w, mode, axis) / w

def test_windowed_mean():
    a = jnp.array([0., 1., 2., 2.])
    assert jnp.allclose(windowed_nanmean(a, 3), jnp.array([2/3., 1., 5/3., 2.]))

def get_coordinates(datapack: DataPack, ref_ant=0, ref_dir=0):
    tmp_selection = datapack._selection
    dummy_soltab = datapack.soltabs[0].replace('000', '')
    datapack.select(ant=ref_ant, dir=ref_dir)
    axes = datapack.__getattr__("axes_{}".format(dummy_soltab))
    _, _ref_ant = datapack.get_antennas(axes['ant'])
    _, _ref_dir = datapack.get_directions(axes['dir'])
    datapack.select(**tmp_selection)
    axes = datapack.__getattr__("axes_{}".format(dummy_soltab))
    _, _antennas = datapack.get_antennas(axes['ant'])
    _, _directions = datapack.get_directions(axes['dir'])
    _, times = datapack.get_times(axes['time'])
    Nt = len(times)

    X_out = []
    ref_ant_out = []
    ref_dir_out = []
    for t in range(Nt):
        obstime = times[t]
        ref_location = ac.ITRS(x=_ref_ant.x, y=_ref_ant.y, z=_ref_ant.z)
        ref_ant = ac.ITRS(x=_ref_ant.x, y=_ref_ant.y, z=_ref_ant.z, obstime=obstime)
        ref_dir = ac.ICRS(ra=_ref_dir.ra, dec=_ref_dir.dec)
        enu = ENU(location=ref_location, obstime=obstime)
        ref_ant = ref_ant.transform_to(enu).cartesian.xyz.to(dist_type).value.T
        ref_dir = ref_dir.transform_to(enu).cartesian.xyz.value.T
        antennas = ac.ITRS(x=_antennas.x, y=_antennas.y, z=_antennas.z, obstime=obstime)
        antennas = antennas.transform_to(enu).cartesian.xyz.to(dist_type).value.T
        directions = ac.ICRS(ra=_directions.ra, dec=_directions.dec)
        directions = directions.transform_to(enu).cartesian.xyz.value.T
        X_out.append(make_coord_array(directions, antennas, flat=False))
        ref_ant_out.append(ref_ant)
        ref_dir_out.append(ref_dir)
    # Nt, Nd, Na, 6
    X = np.stack(X_out, axis=0)
    ref_ant = np.concatenate(ref_ant_out, axis=0)
    ref_dir = np.concatenate(ref_dir_out, axis=0)
    return X, ref_ant, ref_dir

def great_circle_sep(ra1, dec1, ra2, dec2):
    """
    Seperation on S1
    Args:
        ra1: radians
        dec1: radians
        ra2: radians
        dec2: radians

    Returns: radians
    """
    dra = jnp.abs(ra1 - ra2)
    # ddec = np.abs(dec1-dec2)
    num2 = (jnp.cos(dec2) * jnp.sin(dra)) ** 2 + (
            jnp.cos(dec1) * jnp.sin(dec2) - jnp.sin(dec1) * jnp.cos(dec2) * jnp.cos(dra)) ** 2
    den = jnp.sin(dec1) * jnp.sin(dec2) + jnp.cos(dec1) * jnp.cos(dec2) * jnp.cos(dra)
    return jnp.arctan2(jnp.sqrt(num2), den)

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

def test_disable_jit_and_scan():
    from jax import disable_jit
    from jax.lax import scan
    def body(state, X):
        return state, ()

    with disable_jit():
        print(scan(body, (jnp.array(0),), (), length=5))

def get_screen_directions_from_image(image_fits, flux_limit=0.1, max_N=None, min_spacing_arcmin=1., plot=False,
                                     seed_directions=None, fill_in_distance=None,
                                     fill_in_flux_limit=0.):
    """
    Find directions in an apparent flux image satifying a set of selection criteria.

    Args:
        image_fits: FITS image of apparent flux
        flux_limit: float, Selection limit above which to select primary sources.
        max_N: int, Maximum number of sources to find, None is find all above limit.
        min_spacing_arcmin:
        plot:
        seed_directions:
        fill_in_distance: if not None, then search for dimmer sources below `flux_limit` but above `fill_in_flux_limit`
            until we find `max_N` sources within this many arcmin. `fill_in_distance` should be larger than the
            `min_spacing_arcmin`.
        fill_in_flux_limit: the flux limit for the secondary selection criteria.

    Returns:
        ICRS coordinates of directions satisfying the primary and secondary selection criteria.
        Array of radiu in arcsec around the sources beyond which you may subtract sources for peeling.
    """
    logger.info(f"Getting {max_N} screen directions from image {image_fits} with flux above {flux_limit} "
                f"separated by at least {min_spacing_arcmin}.")

    with fits.open(image_fits) as hdul:
        # ra,dec, _, freq
        data = hdul[0].data
        w = wcs.WCS(hdul[0].header)
        #         Nra, Ndec,_,_ = data.shape
        where_limit = np.where(data >= flux_limit)
        arg_sort = np.argsort(data[where_limit])[::-1]

        ra = []
        dec = []
        f = []
        if seed_directions is not None:
            ra = list(seed_directions[:, 0])
            dec = list(seed_directions[:, 1])
            f = list(flux_limit * np.ones(len(ra)))
            logger.info("Using {} seed directions.".format(len(f)))
        idx = []
        for i in arg_sort:
            if max_N is not None:
                if len(ra) >= max_N:
                    break
            pix = [where_limit[3][i], where_limit[2][i], where_limit[1][i], where_limit[0][i]]
            #             logger.info("{} -> {}".format(i, pix))
            #             pix = np.reshape(np.array(np.unravel_index(i, [Nra, Ndec, 1, 1])), (1, 4))
            coords = w.wcs_pix2world([pix], 1)  # degrees
            ra_ = coords[0, 0] * np.pi / 180.
            dec_ = coords[0, 1] * np.pi / 180.

            if len(ra) == 0:
                ra.append(ra_)
                dec.append(dec_)
                f.append(data[pix[3], pix[2], pix[1], pix[0]])
                logger.info(
                    "Auto-append first: Found {} at {} {}".format(f[-1], ra[-1] * 180. / np.pi,
                                                                  dec[-1] * 180. / np.pi))
                idx.append(i)
                continue
            dist = great_circle_sep(np.array(ra), np.array(dec), ra_, dec_) * 180. / np.pi
            if np.all(dist > min_spacing_arcmin / 60.):
                ra.append(ra_)
                dec.append(dec_)
                f.append(data[pix[3], pix[2], pix[1], pix[0]])
                logger.info(
                    "Found source of flux {} at {} {}".format(f[-1], ra[-1] * 180. / np.pi, dec[-1] * 180. / np.pi))
                idx.append(i)

        first_found = len(ra)

        if fill_in_distance is not None:
            logger.info(f"Applying secondary selection criteria. Flux limit {fill_in_flux_limit} and minimum "
                        f"separation {fill_in_distance}.")
            where_limit = np.where(np.logical_and(data < np.min(f), data >= fill_in_flux_limit))
            arg_sort = np.argsort(data[where_limit])[::-1]
            # use remaining brightest sources to get fillers
            for i in arg_sort:
                if max_N is not None:
                    if len(ra) >= max_N:
                        break
                pix = [where_limit[3][i], where_limit[2][i], where_limit[1][i], where_limit[0][i]]
                #                 logger.info("{} -> {}".format(i, pix))
                coords = w.wcs_pix2world([pix], 1)  # degrees

                ra_ = coords[0, 0] * np.pi / 180.
                dec_ = coords[0, 1] * np.pi / 180.

                dist = great_circle_sep(np.array(ra), np.array(dec), ra_, dec_) * 180. / np.pi
                if np.all(dist > fill_in_distance / 60.):
                    ra.append(ra_)
                    dec.append(dec_)
                    f.append(data[pix[3], pix[2], pix[1], pix[0]])
                    logger.info(
                        "Found filler {} at {} {}".format(f[-1], ra[-1] * 180. / np.pi, dec[-1] * 180. / np.pi))
                    idx.append(i)

        second_found = len(ra) - first_found
        total_found = first_found + second_found

        if max_N is not None:
            f = np.array(f)[:total_found]
            ra = np.array(ra)[:total_found]
            dec = np.array(dec)[:total_found]
        sizes = np.ones(len(idx))
        sizes[:first_found] = 120.
        sizes[first_found:] = 240.

    logger.info("Found {} sources in total. {} from primary selection, {} from secondary selection.".format(
        total_found, first_found, second_found))

    return ac.ICRS(ra=ra * au.rad, dec=dec * au.rad), sizes

def inverse_update(C, m, return_drop=False):
    """
    Compute the inverse of a matrix with the m-th row and column dropped given knowledge of the inverse of the original
    matrix.

        C = inv(A)
        B = drop_col(drop_row(A, m),m)
        computes inv(B) given only C

    Args:
        C: inverse of full matirix
        m: row and col to drop
        return_drop: whether to also return the array used to drop the m-th row/col.

    Returns:
        B
        if return_drop:
            the array to drop row/col using jnp.take(v, drop_array)
    """
    drop = drop_array(C.shape[0], m)
    _a = jnp.take(C, drop, axis=0)  # drop m row
    a = jnp.take(_a, drop, axis=1)
    c = jnp.take(C, drop, axis=1)[None, m, :]  # drop m col
    b = _a[:, m, None]
    d = C[m, m]
    res = a - (b @ c) / d
    if return_drop:
        return res, drop
    return res

def test_inverse_update():
    A = np.random.normal(size=(4, 4))
    m = 1
    B = np.linalg.inv(A[np.arange(4) != m, :][:, np.arange(4) != m])
    assert np.isclose(inverse_update(np.linalg.inv(A), m), B).all()

def drop_array(n, m):
    """
    Produce an array of arange(0,n) with the m-th index missing
    Args:
        n: int, maximum index
        m: int, missing index

    Returns:
        an index array suitable to use with jnp.take to take all but the m-th element along an axis.
    """
    # TODO to with mod n
    a = jnp.arange(n)
    a = jnp.roll(a, -m, axis=0)
    a = a[1:]
    a = jnp.roll(a, m, axis=0)
    return a

def polyfit(x, y, deg):
    """
    x : array_like, shape (M,)
        x-coordinates of the M sample points ``(x[i], y[i])``.
    y : array_like, shape (M,) or (M, K)
        y-coordinates of the sample points. Several data sets of sample
        points sharing the same x-coordinates can be fitted at once by
        passing in a 2D-array that contains one dataset per column.
    deg : int
        Degree of the fitting polynomial
    Returns
    -------
    p : ndarray, shape (deg + 1,) or (deg + 1, K)
        Polynomial coefficients, highest power first.  If `y` was 2-D, the
        coefficients for `k`-th data set are in ``p[:,k]``.
    """
    order = int(deg) + 1
    if deg < 0:
        raise ValueError("expected deg >= 0")
    if x.ndim != 1:
        raise TypeError("expected 1D vector for x")
    if x.size == 0:
        raise TypeError("expected non-empty vector for x")
    if y.ndim < 1 or y.ndim > 2:
        raise TypeError("expected 1D or 2D array for y")
    if x.shape[0] != y.shape[0]:
        raise TypeError("expected x and y to have same length")
    rcond = len(x) * jnp.finfo(x.dtype).eps
    lhs = jnp.stack([x ** (deg - i) for i in range(order)], axis=1)
    rhs = y
    scale = jnp.sqrt(jnp.sum(lhs * lhs, axis=0))
    lhs /= scale
    c, resids, rank, s = jnp.linalg.lstsq(lhs, rhs, rcond)
    c = (c.T / scale).T  # broadcast scale coefficients
    return c

def poly_smooth(x, y, deg=5, weights=None):
    """
    Smooth y(x) with a `deg` degree polynomial in x
    Args:
        x: [N]
        y: [N]
        deg: int

    Returns: smoothed y [N]
    """
    if weights is None:
        coeffs = polyfit(x, y, deg=deg)
    else:
        coeffs = weighted_polyfit(x, y, deg=deg, weights=weights)
    return sum([p * x ** (deg - i) for i, p in enumerate(coeffs)])

def weighted_polyfit(x, y, deg, weights):
    order = int(deg) + 1
    if deg < 0:
        raise ValueError("expected deg >= 0")
    if x.ndim != 1:
        raise TypeError("expected 1D vector for x")
    if x.size == 0:
        raise TypeError("expected non-empty vector for x")
    if y.ndim < 1 or y.ndim > 2:
        raise TypeError("expected 1D or 2D array for y")
    if x.shape[0] != y.shape[0]:
        raise TypeError("expected x and y to have same length")
    rcond = len(x) * jnp.finfo(x.dtype).eps
    X = jnp.stack([x ** (deg - i) for i in range(order)], axis=1)
    scale = jnp.sqrt(jnp.sum(X * X, axis=0))
    X /= scale
    c, resids, rank, s = jnp.linalg.lstsq((X.T * weights) @ X, (X.T * weights) @ y, rcond)
    c = (c.T / scale).T  # broadcast scale coefficients
    return c

def test_weighted_polyfit():
    def f(x):
        return 1. + x + x**2 + x**3
    x = jnp.linspace(0., 1., 3)
    assert jnp.allclose(polyfit(x,f(x),4), weighted_polyfit(x,f(x),4, jnp.ones_like(x)),atol=1e-3)

def wrap(phi):
    return (phi + jnp.pi) % (2 * jnp.pi) - jnp.pi

def link_overwrite(src, dst):
    if os.path.islink(dst):
        logger.info("Unlinking pre-existing sym link {}".format(dst))
        os.unlink(dst)
        time.sleep(0.1)  # to make sure it get's removed
    if os.path.abspath(src) != os.path.abspath(dst):
        logger.info("Linking {} -> {}".format(src, dst))
        os.symlink(src, dst)


def curv(Y, times, i):
    """
    Computes the curvature at index i, using a deg=2 polynomial fit to the point and neighbours.
    Args:
        Y: array
        times: array
        i: int

    Returns:
        curvature at times[i]
    """
    _times = jnp.pad(times, ((1, 1),), mode='reflect')
    get_time = lambda i: _times[i + 1]
    _Y = jnp.pad(Y, ((1, 1),), mode='reflect')
    get_Y = lambda i: _Y[i + 1]

    def K_per_offset(offset):
        x = jnp.asarray([get_time(i - 1 + offset), get_time(i + offset), get_time(i + 1 + offset)])
        y = jnp.asarray([get_Y(i - 1 + offset), get_Y(i + offset), get_Y(i + 1 + offset)])

        def f(v):
            coeffs = polyfit(x, y, deg=2)
            return sum([p * v ** (2 - i) for i, p in enumerate(coeffs)])

        fp = grad(f)
        fpp = grad(fp)
        v = x[1 - offset]
        K = jnp.abs(fpp(v)) / (1. + fp(v) ** 2) ** 1.5
        return K

    return jnp.min(vmap(K_per_offset)(jnp.arange(-1, 2)))

def axes_move(array, in_axes, out_axes,size_dict=None):
    """
    Reshape and transpose named axes.

    Args:
        array: ndarray
        in_axes: list of axes. Each element is a formula for the shape, e.g. ['a','bc','d'], where 'bc' is two dimensions flattened in C order.
        out_axes: list of axes. Each element is a formula for the output shape, e.g. ['a', 'b', 'dc'], is a shape where the last is two dimensions flattened in C order. Must contain axes names from in_axes.
        size_dict: optional, helper for the sizes of some axes. dict(a=3) would enforce that axes 'a' is size 3.

    Returns:
        array is reshaped from in_axes to an expanded form, permutated to the correct order, and then reshaped to out_axes.

    Raises:
        ValueError if shape solution not possible.
        ValueError if out_axes contains axes not in in_axes
    """
    if size_dict is None:
        size_dict = dict()
    _in_axes = "".join(in_axes)
    _out_axes = "".join(out_axes)
    if set(_in_axes) != set(_out_axes):
        raise ValueError(f"in_axes {in_axes} should have all the same symbols as out_axes {out_axes}")
    symbols = {dim: sympy.symbols(dim) for dim in set(list(_in_axes)+list(_out_axes)+list(size_dict.keys()))}
    def _eq(eq, size):
        output = sympy.Rational(1)
        for dim in eq:
            output *= symbols[dim]
        return sympy.Eq(output, sympy.Rational(size))
    # print([_eq(eq, size) for (eq, size) in zip(in_axes, array.shape)] + [_eq(dim, size) for dim, size in size_dict.items()])
    sol = sympy.solve([_eq(eq, size) for (eq, size) in zip(in_axes, array.shape)] + [_eq(dim, size) for dim, size in size_dict.items()],
                      # symbols.values(),
                      dict=True)[0]
    for sym in symbols.values():
        if sym not in sol.keys():
            raise ValueError(f"Not enough information to solve for shape {sym}. Solution is {sol}.")
    size_dict = {dim:sol[sym] for dim,sym in symbols.items()}


    array = array.reshape([size_dict[dim] for dim in _in_axes])
    perm = [_in_axes.index(d) for d in _out_axes]
    array = array.transpose(perm)
    array = array.reshape([np.prod([size_dict[dim] for dim in dim_prod]) for dim_prod in out_axes])
    return array

def test_axes_move():
    array = jnp.ones((1,2,3,4))
    _array = axes_move(array, ['a','b','c','d'], ['d','b','c','a'],size_dict=None)
    assert _array.shape == (4,2,3,1)

    _array = axes_move(array, ['a', 'b', 'c', 'd'], ['db', 'c', 'a'],size_dict=None)
    assert _array.shape == (4*2, 3, 1)

    _array = axes_move(array, ['a', 'b', 'c', 'd'], ['c', 'db', 'a'],size_dict=None)
    assert _array.shape == (3, 4 * 2, 1)

    _array = axes_move(array, ['a', 'b', 'c', 'de'], ['c', 'db', 'a', 'e'], size_dict=dict(e=2))
    assert _array.shape == (3, 2 * 2, 1, 2)