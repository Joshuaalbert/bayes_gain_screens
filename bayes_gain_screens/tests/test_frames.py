from .common_setup import *

import numpy as np
from astropy import time as at, coordinates as ac, units as au

from bayes_filter import dist_type
from bayes_filter.frames import ENU


def test_enu(lofar_array):
    antennas = lofar_array[1]
    obstime = at.Time("2018-01-01T00:00:00.000", format='isot')
    location = ac.ITRS(x=antennas[0,0] * dist_type, y=antennas[0,1] * dist_type, z=antennas[0,2] * dist_type)
    enu = ENU(obstime=obstime,location=location.earth_location)
    altaz = ac.AltAz(obstime=obstime,location=location.earth_location)
    lofar_antennas = ac.ITRS(x=antennas[:, 0] * dist_type, y=antennas[:, 1] * dist_type, z=antennas[:, 2] * dist_type, obstime=obstime)
    assert np.all(np.linalg.norm(lofar_antennas.transform_to(enu).cartesian.xyz.to(dist_type).value, axis=0) < 100.)
    assert np.all(np.isclose(lofar_antennas.transform_to(enu).cartesian.xyz.to(dist_type).value,
                             lofar_antennas.transform_to(enu).transform_to(altaz).transform_to(enu).cartesian.xyz.to(
                                 dist_type).value))
    assert np.all(np.isclose(lofar_antennas.transform_to(altaz).cartesian.xyz.to(dist_type).value,
                             lofar_antennas.transform_to(altaz).transform_to(enu).transform_to(altaz).cartesian.xyz.to(
                                 dist_type).value))
    north_enu = ac.SkyCoord(east=0.,north=1., up=0.,frame=enu)
    north_altaz = ac.SkyCoord(az=0*au.deg, alt=0*au.deg, distance=1.,frame=altaz)
    assert np.all(np.isclose(
        north_enu.transform_to(altaz).cartesian.xyz.value, north_altaz.cartesian.xyz.value))
    assert np.all(np.isclose(
        north_enu.cartesian.xyz.value, north_altaz.transform_to(enu).cartesian.xyz.value))
    east_enu = ac.SkyCoord(east=1., north=0., up=0., frame=enu)
    east_altaz = ac.SkyCoord(az=90 * au.deg, alt=0 * au.deg, distance=1., frame=altaz)
    assert np.all(np.isclose(
        east_enu.transform_to(altaz).cartesian.xyz.value, east_altaz.cartesian.xyz.value))
    assert np.all(np.isclose(
        east_enu.cartesian.xyz.value, east_altaz.transform_to(enu).cartesian.xyz.value))
    up_enu = ac.SkyCoord(east=0., north=0., up=1., frame=enu)
    up_altaz = ac.SkyCoord(az=0 * au.deg, alt=90 * au.deg, distance=1., frame=altaz)
    assert np.all(np.isclose(
        up_enu.transform_to(altaz).cartesian.xyz.value, up_altaz.cartesian.xyz.value))
    assert np.all(np.isclose(
        up_enu.cartesian.xyz.value, up_altaz.transform_to(enu).cartesian.xyz.value))
    ###
    # dimensionful
    north_enu = ac.SkyCoord(east=0.*dist_type, north=1.*dist_type, up=0.*dist_type, frame=enu)
    north_altaz = ac.SkyCoord(az=0 * au.deg, alt=0 * au.deg, distance=1.*dist_type, frame=altaz)
    assert np.all(np.isclose(
        north_enu.transform_to(altaz).cartesian.xyz.to(dist_type).value, north_altaz.cartesian.xyz.to(dist_type).value))
    assert np.all(np.isclose(
        north_enu.cartesian.xyz.to(dist_type).value, north_altaz.transform_to(enu).cartesian.xyz.to(dist_type).value))
    east_enu = ac.SkyCoord(east=1.*dist_type, north=0.*dist_type, up=0.*dist_type, frame=enu)
    east_altaz = ac.SkyCoord(az=90 * au.deg, alt=0 * au.deg, distance=1.*dist_type, frame=altaz)
    assert np.all(np.isclose(
        east_enu.transform_to(altaz).cartesian.xyz.to(dist_type).value, east_altaz.cartesian.xyz.to(dist_type).value))
    assert np.all(np.isclose(
        east_enu.cartesian.xyz.to(dist_type).value, east_altaz.transform_to(enu).cartesian.xyz.to(dist_type).value))
    up_enu = ac.SkyCoord(east=0.*dist_type, north=0.*dist_type, up=1.*dist_type, frame=enu)
    up_altaz = ac.SkyCoord(az=0 * au.deg, alt=90 * au.deg, distance=1.*dist_type, frame=altaz)
    assert np.all(np.isclose(
        up_enu.transform_to(altaz).cartesian.xyz.to(dist_type).value, up_altaz.cartesian.xyz.to(dist_type).value))
    assert np.all(np.isclose(
        up_enu.cartesian.xyz.to(dist_type).value, up_altaz.transform_to(enu).cartesian.xyz.to(dist_type).value))