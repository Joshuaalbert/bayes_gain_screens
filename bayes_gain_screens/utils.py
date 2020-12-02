from timeit import default_timer

import numpy as np
import astropy.coordinates as ac
import astropy.units as au
from astropy import wcs
from astropy.io import fits

from jax import local_device_count, tree_map, numpy as jnp, pmap
from jax.lax import scan

from bayes_gain_screens.frames import ENU

from h5parm import DataPack

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


def rolling_window(a, window, padding='same'):
    if padding.lower() == 'same':
        pad_start = np.zeros(len(a.shape), dtype=np.int32)
        pad_start[-1] = window // 2
        pad_end = np.zeros(len(a.shape), dtype=np.int32)
        pad_end[-1] = (window - 1) - pad_start[-1]
        pad = list(zip(pad_start, pad_end))
        a = np.pad(a, pad, mode='reflect')
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def apply_rolling_func_strided(func, a, window, piecewise_constant=True):
    if not piecewise_constant:
        rolling_a = rolling_window(a, window, padding='same')
        return func(rolling_a)
    # handles case when window larger
    window = min(a.shape[-1], window)
    # ..., N//window, window
    rolling_a = rolling_window(a, window, padding='valid')[..., ::window, :]
    repeat = a.shape[-1] // rolling_a.shape[-2]
    # ..., N//window
    b = func(rolling_a)
    repeats = repeat * np.ones(b.shape[-1], dtype=np.int64)
    repeats[-1] = a.shape[-1] - (repeat * (b.shape[-1] - 1))
    # ...,N (hopefully)
    return np.repeat(b, repeats, axis=-1)


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


def chunked_pmap(f, *args, chunksize=None):
    """
    Calls pmap on chunks of moderate work to be distributed over devices.
    Automatically handle non-dividing chunksizes, by adding filler elements.

    Args:
        f: callable
        *args: arguments to map down first dimension
        chunksize: optional chunk size else num devices

    Returns: pytree mapped result.
    """
    # if chunksize is None:
    chunksize = local_device_count()
    N = args[0].shape[0]
    remainder = N % chunksize
    if (remainder != 0) and (N > chunksize):
        extra = chunksize - remainder
        args = tree_map(lambda arg: jnp.concatenate([arg, arg[:extra]], axis=0), args)
        N = args[0].shape[0]
    args = tree_map(lambda arg: jnp.reshape(arg, (chunksize, N // chunksize) + arg.shape[1:]), args)
    logger.info("Running on {}".format(chunksize))
    t0 = default_timer()

    def pmap_body(*args):
        def body(state, args):
            return state, f(*args)

        _, result = scan(body, (), args, unroll=1)
        return result

    result = pmap(pmap_body)(*args)
    result = tree_map(lambda arg: jnp.reshape(arg, (-1,) + arg.shape[2:]), result)
    if remainder != 0:
        result = tree_map(lambda x: x[:-extra], result)
    dt = default_timer() - t0
    logger.info("Time to run: {}, rate: {} / s".format(dt, N / dt))
    return result

def get_screen_directions_from_image(image_fits, flux_limit=0.1, max_N=None, min_spacing_arcmin=1., plot=False,
                                     seed_directions=None, fill_in_distance=None,
                                     fill_in_flux_limit=0.):
    """Given a srl file containing the sources extracted from the apparent flux image of the field,
    decide the screen directions
    :param srl_fits: str
        The path to the srl file, typically created by pybdsf
    :return: float, array [N, 2]
        The `N` sources' coordinates as an ``astropy.coordinates.ICRS`` object
    """
    logging.info("Getting screen directions from image.")

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
            logging.info("Using seed directions.")
            ra = list(seed_directions[:, 0])
            dec = list(seed_directions[:, 1])
            f = list(flux_limit * np.ones(len(ra)))
        idx = []
        for i in arg_sort:
            if max_N is not None:
                if len(ra) >= max_N:
                    break
            pix = [where_limit[3][i], where_limit[2][i], where_limit[1][i], where_limit[0][i]]
            #             logging.info("{} -> {}".format(i, pix))
            #             pix = np.reshape(np.array(np.unravel_index(i, [Nra, Ndec, 1, 1])), (1, 4))
            coords = w.wcs_pix2world([pix], 1)  # degrees
            ra_ = coords[0, 0] * np.pi / 180.
            dec_ = coords[0, 1] * np.pi / 180.

            if len(ra) == 0:
                ra.append(ra_)
                dec.append(dec_)
                f.append(data[pix[3], pix[2], pix[1], pix[0]])
                logging.info(
                    "Auto-append first: Found {} at {} {}".format(f[-1], ra[-1] * 180. / np.pi, dec[-1] * 180. / np.pi))

                idx.append(i)
                continue
            dist = great_circle_sep(np.array(ra), np.array(dec), ra_, dec_) * 180. / np.pi
            if np.all(dist > min_spacing_arcmin / 60.):
                ra.append(ra_)
                dec.append(dec_)
                f.append(data[pix[3], pix[2], pix[1], pix[0]])
                logging.info("Found {} at {} {}".format(f[-1], ra[-1] * 180. / np.pi, dec[-1] * 180. / np.pi))
                idx.append(i)

        first_found = len(idx)

        if fill_in_distance is not None:
            where_limit = np.where(np.logical_and(data < np.min(f), data >= fill_in_flux_limit))
            arg_sort = np.argsort(data[where_limit])[::-1]
            # use remaining brightest sources to get fillers
            for i in arg_sort:
                if max_N is not None:
                    if len(ra) >= max_N:
                        break
                pix = [where_limit[3][i], where_limit[2][i], where_limit[1][i], where_limit[0][i]]
                #                 logging.info("{} -> {}".format(i, pix))
                coords = w.wcs_pix2world([pix], 1)  # degrees

                ra_ = coords[0, 0] * np.pi / 180.
                dec_ = coords[0, 1] * np.pi / 180.

                dist = great_circle_sep(np.array(ra), np.array(dec), ra_, dec_) * 180. / np.pi
                if np.all(dist > fill_in_distance / 60.):
                    ra.append(ra_)
                    dec.append(dec_)
                    f.append(data[pix[3], pix[2], pix[1], pix[0]])
                    logging.info(
                        "Found filler {} at {} {}".format(f[-1], ra[-1] * 180. / np.pi, dec[-1] * 180. / np.pi))
                    idx.append(i)

        if max_N is not None:
            f = np.array(f)[:max_N]
            ra = np.array(ra)[:max_N]
            dec = np.array(dec)[:max_N]
        sizes = np.ones(len(idx))
        sizes[:first_found] = 120.
        sizes[first_found:] = 240.

    logging.info("Found {} sources.".format(len(ra)))

    return ac.ICRS(ra=ra * au.rad, dec=dec * au.rad), sizes