import os
from timeit import default_timer

import numpy as np
import astropy.coordinates as ac
import astropy.units as au
from astropy import wcs
from astropy.io import fits

from jax import local_device_count, tree_map, numpy as jnp, pmap, vmap, jit, tree_multimap, devices as get_devices, \
    device_get
from jax.lax import scan
from jax.scipy.signal import convolve
from jax.api import _jit_is_disabled as jit_is_disabled

from bayes_gain_screens.frames import ENU

from h5parm import DataPack

from dask.threaded import get

import logging

logger = logging.getLogger(__name__)
dist_type = au.km
angle_type = au.rad


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
    if axis != 0:
        a = jnp.moveaxis(a, axis, 0)
    if w is None:
        return jnp.broadcast_to(jnp.mean(a, axis=0, keepdims=True), a.shape)
    dims = len(a.shape)
    a = a
    kernel = jnp.reshape(jnp.ones(w) / w, [w] + [1] * (dims - 1))
    _w1 = (w - 1) // 2
    _w2 = _w1 if (w % 2 == 1) else _w1 + 1
    pad_width = [(_w1, _w2)] + [(0, 0)] * (dims - 1)
    a = jnp.pad(a, pad_width=pad_width, mode=mode)
    result = convolve(a, kernel, mode='valid', precision=None)
    if axis != 0:
        result = jnp.moveaxis(result, 0, axis)
    return result


def test_windowed_mean():
    a = jnp.arange(10)
    a = jnp.where(jnp.arange(10) >= 8, jnp.nan, jnp.arange(10))
    assert jnp.all(jnp.isnan(a) == jnp.isnan(windowed_mean(a, 1)))


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
    dra = np.abs(ra1 - ra2)
    # ddec = np.abs(dec1-dec2)
    num2 = (np.cos(dec2) * np.sin(dra)) ** 2 + (
            np.cos(dec1) * np.sin(dec2) - np.sin(dec1) * np.cos(dec2) * np.cos(dra)) ** 2
    den = np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(dra)
    return np.arctan2(np.sqrt(num2), den)


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


def chunked_pmap(f, *args, chunksize=None, debug_mode=False):
    """
    Calls pmap on chunks of moderate work to be distributed over devices.
    Automatically handle non-dividing chunksizes, by adding filler elements.

    Args:
        f: jittable, callable
        *args: ndarray arguments to map down first dimension
        chunksize: optional chunk size, should be <= local_device_count(). None is local_device_count.
        debug_mode: bool, whether to allow printing but also jit by using dask

    Returns: pytree mapped result.
    """
    if chunksize is None:
        chunksize = local_device_count()
    if chunksize > local_device_count():
        raise ValueError("chunksize should be <= {}".format(local_device_count()))
    N = args[0].shape[0]
    remainder = N % chunksize
    if (remainder != 0) and (N > chunksize):
        extra = chunksize - remainder
        args = tree_map(lambda arg: jnp.concatenate([arg, arg[:extra]], axis=0), args)
        N = args[0].shape[0]
    args = tree_map(lambda arg: jnp.reshape(arg, (chunksize, N // chunksize) + arg.shape[1:]), args)
    T = N // chunksize

    logger.info("Distributing {} over {}".format(N, chunksize))
    t0 = default_timer()

    if debug_mode:
        from datetime import datetime
        devices = get_devices()
        def build_pmap_body(dev_idx):
            fun = jit(f, device=devices[dev_idx])
            log = os.path.join(os.getcwd(),"chunk{:02d}.log".format(dev_idx))
            def pmap_body(*args):
                result = []
                with open(log, 'a') as f:
                    for i in range(T):
                        item = jnp.ravel_multi_index((dev_idx, i), (chunksize, T))
                        logger.info("Starting item: {}".format(item))
                        f.write('{} {}'.format(datetime.now().isoformat(), "Starting item: {}".format(item)))
                        result.append(fun(*[a[i, ...] for a in args]))
                        tree_map(lambda a: a.block_until_ready(), result[-1])
                        logger.info("Done item: {}".format(item))
                        f.write('{} {}'.format(datetime.now().isoformat(), "Done item: {}".format(item)))
                result = tree_multimap(lambda *result: jnp.stack(result, axis=0), *result)
                return result
            return pmap_body
    else:
        @jit
        def pmap_body(*args):
            def body(state, args):
                return state, f(*args)
            _, result = scan(body, (), args, unroll=1)
            return result

    if jit_is_disabled():
        if debug_mode:
            num_devices = local_device_count()
            dsk = {str(device): (build_pmap_body(device),) + tuple([arg[device] for arg in args]) for device in
                   range(num_devices)}
            result_keys = [str(device) for device in range(num_devices)]
            result = get(dsk, result_keys, num_workers=num_devices)
            result = device_get(result)
            result = tree_multimap(lambda *result: jnp.stack(result, axis=0), *result)
        else:
            result = vmap(pmap_body)(*args)
    else:
        if debug_mode:
            num_devices = local_device_count()
            dsk = {str(device): (build_pmap_body(device),) + tuple([arg[device] for arg in args]) for device in
                   range(num_devices)}
            result_keys = [str(device) for device in range(num_devices)]
            result = get(dsk, result_keys, num_workers=num_devices)
            result = device_get(result)
            result = tree_multimap(lambda *result: jnp.stack(result, axis=0), *result)
        else:
            result = pmap(pmap_body)(*args)  # until gh #5065 is solved

    result = tree_map(lambda arg: jnp.reshape(arg, (-1,) + arg.shape[2:]), result)
    if remainder != 0:
        result = tree_map(lambda x: x[:-extra], result)
    dt = default_timer() - t0
    logger.info("Time to run: {}, rate: {} / s, normalised rate: {} / s / device".format(dt, N / dt, N / dt / chunksize))
    return result


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
                    "Auto-append first: Found {} at {} {}".format(f[-1], ra[-1] * 180. / np.pi, dec[-1] * 180. / np.pi))
                idx.append(i)
                continue
            dist = great_circle_sep(np.array(ra), np.array(dec), ra_, dec_) * 180. / np.pi
            if np.all(dist > min_spacing_arcmin / 60.):
                ra.append(ra_)
                dec.append(dec_)
                f.append(data[pix[3], pix[2], pix[1], pix[0]])
                logger.info("Found source of flux {} at {} {}".format(f[-1], ra[-1] * 180. / np.pi, dec[-1] * 180. / np.pi))
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


def poly_smooth(x, y, deg=5):
    """
    Smooth y(x) with a `deg` degree polynomial in x
    Args:
        x: [N]
        y: [N]
        deg: int

    Returns: smoothed y [N]
    """
    coeffs = polyfit(x, y, deg=deg)
    return sum([p * x ** (deg - i) for i, p in enumerate(coeffs)])


def wrap(phi):
    return (phi + jnp.pi) % (2 * jnp.pi) - jnp.pi


def link_overwrite(src, dst):
    if os.path.islink(dst):
        logger.info("Unlinking pre-existing sym link {}".format(dst))
        os.unlink(dst)
    logger.info("Linking {} -> {}".format(src, dst))
    os.symlink(src, dst)