

from __future__ import (absolute_import, unicode_literals, division, print_function)
import numpy as np
import astropy.units as u
from astropy.coordinates.baseframe import (BaseCoordinateFrame, RepresentationMapping, frame_transform_graph)
from astropy.coordinates.transformations import FunctionTransform
from astropy.coordinates.representation import (SphericalRepresentation, UnitSphericalRepresentation,
                                                CartesianRepresentation)
from astropy.coordinates.attributes import (TimeAttribute, CoordinateAttribute, EarthLocationAttribute)
from astropy.coordinates import ITRS, ICRS, AltAz


class ENU(BaseCoordinateFrame):
    """
    Written by Joshua G. Albert - albert@strw.leidenuniv.nl
    A coordinate or frame in the East-North-Up (ENU) system.
    This frame has the following frame attributes, which are necessary for
    transforming from ENU to some other system:
    * ``obstime``
        The time at which the observation is taken.  Used for determining the
        position and orientation of the Earth.
    * ``location``
        The location on the Earth.  This can be specified either as an
        `~astropy.coordinates.EarthLocation` object or as anything that can be
        transformed to an `~astropy.coordinates.ITRS` frame.
    Parameters
    ----------
    representation : `BaseRepresentation` or None
        A representation object or None to have no data (or use the other keywords)
    east : :class:`~astropy.units.Quantity`, optional, must be keyword
        The east coordinate for this object (``north`` and ``up`` must also be given and
        ``representation`` must be None).
    north : :class:`~astropy.units.Quantity`, optional, must be keyword
        The north coordinate for this object (``east`` and ``up`` must also be given and
        ``representation`` must be None).
    up : :class:`~astropy.units.Quantity`, optional, must be keyword
        The up coordinate for this object (``north`` and ``east`` must also be given and
        ``representation`` must be None).
    Notes
    -----
    This is useful as an intermediate frame between ITRS and UVW for radio astronomy
    """

    frame_specific_representation_info = {
        'cartesian': [RepresentationMapping('x', 'east'),
                      RepresentationMapping('y', 'north'),
                      RepresentationMapping('z', 'up')],
    }

    default_representation = CartesianRepresentation

    obstime = TimeAttribute(default=None)  # at.Time("2000-01-01T00:00:00.000",format="isot",scale="tai"))
    location = EarthLocationAttribute(default=None)

    def __init__(self, *args, **kwargs):
        super(ENU, self).__init__(*args, **kwargs)


@frame_transform_graph.transform(FunctionTransform, AltAz, ENU)
def altaz_to_enu(altaz_coo, enu_frame):
    '''Defines the transformation between AltAz and the ENU frame.
    AltAz usually has units attached but ENU does not require units
    if it specifies a direction.'''
    is_directional = (isinstance(altaz_coo.data, UnitSphericalRepresentation) or
                      altaz_coo.cartesian.x.unit == u.one)

    if is_directional:
        rep = CartesianRepresentation(x=altaz_coo.cartesian.y,
                                      y=altaz_coo.cartesian.x,
                                      z=altaz_coo.cartesian.z,
                                      copy=False)
    else:
        # location_altaz = ITRS(*enu_frame.location.to_geocentric()).transform_to(
        #     AltAz(location=enu_frame.location, obstime=enu_frame.obstime))
        rep = CartesianRepresentation(x=altaz_coo.cartesian.y,# - location_altaz.cartesian.y,
                                      y=altaz_coo.cartesian.x,# - location_altaz.cartesian.x,
                                      z=altaz_coo.cartesian.z,# - location_altaz.cartesian.z,
                                      copy=False)
    return enu_frame.realize_frame(rep)


@frame_transform_graph.transform(FunctionTransform, ENU, AltAz)
def enu_to_altaz(enu_coo, altaz_frame):
    is_directional = (isinstance(enu_coo.data, UnitSphericalRepresentation) or
                      enu_coo.cartesian.x.unit == u.one)

    if is_directional:
        rep = CartesianRepresentation(x=enu_coo.north,
                                      y=enu_coo.east,
                                      z=enu_coo.up,
                                      copy=False)
    else:
        # location_altaz = ITRS(*enu_coo.location.to_geocentric()).transform_to(
        #     AltAz(location=enu_coo.location, obstime=enu_coo.obstime))
        rep = CartesianRepresentation(x=enu_coo.north,# + location_altaz.cartesian.x,
                                      y=enu_coo.east,# + location_altaz.cartesian.y,
                                      z=enu_coo.up,# + location_altaz.cartesian.z,
                                      copy=False)
    return altaz_frame.realize_frame(rep)


@frame_transform_graph.transform(FunctionTransform, ENU, ENU)
def enu_to_enu(from_coo, to_frame):
    # for now we just implement this through AltAz to make sure we get everything
    # covered
    return from_coo.transform_to(AltAz(location=from_coo.location, obstime=from_coo.obstime)).transform_to(to_frame)


# class Pointing(BaseCoordinateFrame):
#     """
#     Written by Joshua G. Albert - albert@strw.leidenuniv.nl
#     A coordinate or frame in the Pointing system.
#     This frame has the following frame attributes, which are necessary for
#     transforming from Pointing to some other system:
#     * ``obstime``
#         The time at which the observation is taken.  Used for determining the
#         position and orientation of the Earth.
#     * ``location``
#         The location on the Earth.  This can be specified either as an
#         `~astropy.coordinates.EarthLocation` object or as anything that can be
#         transformed to an `~astropy.coordinates.ITRS` frame.
#     * ``phaseDir``
#         The phase tracking center of the frame.  This can be specified either as an
#         (ra,dec) `~astropy.units.Qunatity` or as anything that can be
#         transformed to an `~astropy.coordinates.ICRS` frame.
#     * ``fixtime``
#         The time at which the frame is fixed.  Used for determining the
#         position and orientation of the Earth.
#     Parameters
#     ----------
#     representation : `BaseRepresentation` or None
#         A representation object or None to have no data (or use the other keywords)
#     u : :class:`~astropy.units.Quantity`, optional, must be keyword
#         The u coordinate for this object (``v`` and ``w`` must also be given and
#         ``representation`` must be None).
#     v : :class:`~astropy.units.Quantity`, optional, must be keyword
#         The v coordinate for this object (``u`` and ``w`` must also be given and
#         ``representation`` must be None).
#     w : :class:`~astropy.units.Quantity`, optional, must be keyword
#         The w coordinate for this object (``u`` and ``v`` must also be given and
#         ``representation`` must be None).
#     Notes
#     -----
#     This is useful for radio astronomy.
#     """
#
#     frame_specific_representation_info = {
#         'cartesian': [RepresentationMapping('x', 'u'),
#                       RepresentationMapping('y', 'v'),
#                       RepresentationMapping('z', 'w')],
#     }
#
#     default_representation = CartesianRepresentation
#
#     obstime = TimeAttribute(default=None)
#     location = EarthLocationAttribute(default=None)
#     phase = CoordinateAttribute(ICRS, default=None)
#     fixtime = TimeAttribute(default=None)
#
#     def __init__(self, *args, **kwargs):
#         super(Pointing, self).__init__(*args, **kwargs)
#
#     @property
#     def elevation(self):
#         """
#         Elevation above the horizon of the direction, in degree
#         """
#         return self.phase.transform_to(AltAz(location=self.location, obstime=self.obstime)).alt
#
#
# @frame_transform_graph.transform(FunctionTransform, ITRS, Pointing)
# def itrs_to_pointing(itrs_coo, pointing_frame):
#     '''Defines the transformation between ITRS and the Pointing frame.'''
#
#     # if np.any(itrs_coo.obstime != pointing_frame.obstime):
#     #    itrs_coo = itrs_coo.transform_to(ITRS(obstime=pointing_frame.obstime))
#
#     # if the data are UnitSphericalRepresentation, we can skip the distance calculations
#     is_unitspherical = (isinstance(itrs_coo.data, UnitSphericalRepresentation) or
#                         itrs_coo.cartesian.x.unit == u.one)
#     # 'WGS84'
#     lon, lat, height = pointing_frame.location.to_geodetic('WGS84')
#     lst = pointing_frame.obstime.sidereal_time('mean',
#                                                lon)  # AltAz(alt=90*u.deg,az=0*u.deg,location=pointing_frame.location,obstime=pointing_frame.fixtime).transform_to(ICRS).ra
#     ha = (lst - pointing_frame.phase.ra).to(u.radian).value
#     dec = pointing_frame.phase.dec.to(u.radian).value
#     lonrad = lon.to(u.radian).value - ha
#     latrad = dec  # lat.to(u.radian).value +
#     sinlat = np.sin(latrad)
#     coslat = np.cos(latrad)
#     sinlon = np.sin(lonrad)
#     coslon = np.cos(lonrad)
#     north = [-sinlat * coslon,
#              -sinlat * sinlon,
#              coslat]
#     east = [-sinlon, coslon, 0]
#     up = [coslat * coslon, coslat * sinlon, sinlat]
#     R = np.array([east, north, up])
#
#     if is_unitspherical:
#         # don't need to do distance calculation
#         p = itrs_coo.cartesian.xyz.value
#         diff = p
#         penu = R.dot(diff)
#
#         rep = CartesianRepresentation(x=u.Quantity(penu[0], u.one, copy=False),
#                                       y=u.Quantity(penu[1], u.one, copy=False),
#                                       z=u.Quantity(penu[2], u.one, copy=False),
#                                       copy=False)
#     else:
#         p = itrs_coo.cartesian.xyz
#         # ,obstime=pointing_frame.fixtime
#         p0 = ITRS(*pointing_frame.location.geocentric).cartesian.xyz
#         diff = (p.T - p0).T
#         penu = R.dot(diff)
#
#         rep = CartesianRepresentation(x=penu[0],  # u.Quantity(penu[0],u.m,copy=False),
#                                       y=penu[1],  # u.Quantity(penu[1],u.m,copy=False),
#                                       z=penu[2],  # u.Quantity(penu[2],u.m,copy=False),
#                                       copy=False)
#
#     return pointing_frame.realize_frame(rep)
#
#
# @frame_transform_graph.transform(FunctionTransform, Pointing, ITRS)
# def pointing_to_itrs(pointing_coo, itrs_frame):
#     # p = itrs_frame.cartesian.xyz.to(u.m).value
#     # p0 = np.array(enu_coo.location.to(u.m).value)
#     # p = np.array(itrs_frame.location.to(u.m).value)
#
#     #
#     lon, lat, height = pointing_coo.location.to_geodetic('WGS84')
#     lst = pointing_coo.obstime.sidereal_time('mean',
#                                              lon)  # lst = AltAz(alt=90*u.deg,az=0*u.deg,location=pointing_coo.location,obstime=pointing_coo.fixtime).transform_to(ICRS).ra
#     ha = (lst - pointing_coo.phase.ra).to(u.radian).value
#     dec = pointing_coo.phase.dec.to(u.radian).value
#     lonrad = lon.to(u.radian).value - ha
#     latrad = dec  # lat.to(u.radian).value +
#     sinlat = np.sin(latrad)
#     coslat = np.cos(latrad)
#     sinlon = np.sin(lonrad)
#     coslon = np.cos(lonrad)
#
#     north = [-sinlat * coslon,
#              -sinlat * sinlon,
#              coslat]
#     east = [-sinlon, coslon, 0]
#     up = [coslat * coslon, coslat * sinlon, sinlat]
#     R = np.array([east, north, up])
#
#     if isinstance(pointing_coo.data, UnitSphericalRepresentation) or pointing_coo.cartesian.x.unit == u.one:
#         diff = R.T.dot(pointing_coo.cartesian.xyz)
#         p = diff
#         rep = CartesianRepresentation(x=u.Quantity(p[0], u.one, copy=False),
#                                       y=u.Quantity(p[1], u.one, copy=False),
#                                       z=u.Quantity(p[2], u.one, copy=False),
#                                       copy=False)
#     else:
#         diff = R.T.dot(pointing_coo.cartesian.xyz)
#         # ,obstime=pointing_coo.fixtime
#         p0 = ITRS(*pointing_coo.location.geocentric).cartesian.xyz
#         # print (R,diff)
#         p = (diff.T + p0).T
#         # print (p)
#         rep = CartesianRepresentation(x=p[0],  # u.Quantity(p[0],u.m,copy=False),
#                                       y=p[1],  # u.Quantity(p[1],u.m,copy=False),
#                                       z=p[2],  # u.Quantity(p[2],u.m,copy=False),
#                                       copy=False)
#
#     return itrs_frame.realize_frame(rep)
#
#     # return ITRS(*p*u.m,obstime=enu_coo.obstime).transform_to(itrs_frame)
#
#
# @frame_transform_graph.transform(FunctionTransform, Pointing, Pointing)
# def pointing_to_pointing(from_coo, to_frame):
#     # for now we just implement this through ITRS to make sure we get everything
#     # covered
#     return from_coo.transform_to(ITRS(obstime=from_coo.obstime)).transform_to(to_frame)
#
#
# #    return from_coo.transform_to(ITRS(obstime=from_coo.obstime,location=from_coo.location)).transform_to(to_frame)
#
#
# class UVW(BaseCoordinateFrame):
#     """
#     Written by Joshua G. Albert - albert@strw.leidenuniv.nl
#     A coordinate or frame in the UVW system.
#     This frame has the following frame attributes, which are necessary for
#     transforming from UVW to some other system:
#     * ``obstime``
#         The time at which the observation is taken.  Used for determining the
#         position and orientation of the Earth.
#     * ``location``
#         The location on the Earth.  This can be specified either as an
#         `~astropy.coordinates.EarthLocation` object or as anything that can be
#         transformed to an `~astropy.coordinates.ITRS` frame.
#     * ``phaseDir``
#         The phase tracking center of the frame.  This can be specified either as an
#         (ra,dec) `~astropy.units.Qunatity` or as anything that can be
#         transformed to an `~astropy.coordinates.ICRS` frame.
#     Parameters
#     ----------
#     representation : `BaseRepresentation` or None
#         A representation object or None to have no data (or use the other keywords)
#     u : :class:`~astropy.units.Quantity`, optional, must be keyword
#         The u coordinate for this object (``v`` and ``w`` must also be given and
#         ``representation`` must be None).
#     v : :class:`~astropy.units.Quantity`, optional, must be keyword
#         The v coordinate for this object (``u`` and ``w`` must also be given and
#         ``representation`` must be None).
#     w : :class:`~astropy.units.Quantity`, optional, must be keyword
#         The w coordinate for this object (``u`` and ``v`` must also be given and
#         ``representation`` must be None).
#     Notes
#     -----
#     This is useful for radio astronomy.
#     """
#
#     frame_specific_representation_info = {
#         'cartesian': [RepresentationMapping('x', 'u'),
#                       RepresentationMapping('y', 'v'),
#                       RepresentationMapping('z', 'w')],
#     }
#
#     default_representation = CartesianRepresentation
#
#     obstime = TimeAttribute(default=None)
#     location = EarthLocationAttribute(default=None)
#     phase = CoordinateAttribute(ICRS, default=None)
#
#     def __init__(self, *args, **kwargs):
#         super(UVW, self).__init__(*args, **kwargs)
#
#     @property
#     def elevation(self):
#         """
#         Elevation above the horizon of the direction
#         """
#         return self.phase.transform_to(AltAz(location=self.location, obstime=self.obstime)).alt
#
#
# @frame_transform_graph.transform(FunctionTransform, ITRS, UVW)
# def itrs_to_uvw(itrs_coo, uvw_frame):
#     '''Defines the transformation between ITRS and the UVW frame.'''
#
#     # if np.any(itrs_coo.obstime != uvw_frame.obstime):
#     #    itrs_coo = itrs_coo.transform_to(ITRS(obstime=uvw_frame.obstime))
#
#     # if the data are UnitSphericalRepresentation, we can skip the distance calculations
#     is_unitspherical = (isinstance(itrs_coo.data, UnitSphericalRepresentation) or
#                         itrs_coo.cartesian.x.unit == u.one)
#
#     lon, lat, height = uvw_frame.location.to_geodetic('WGS84')
#     lst_ = AltAz(alt=90 * u.deg, az=0 * u.deg, location=uvw_frame.location, obstime=uvw_frame.obstime).transform_to(
#         ICRS).ra
#     lst = uvw_frame.obstime.sidereal_time('apparent', lon)
#     # print(lst.deg,lst_.deg)
#     ha = (lst - uvw_frame.phase.ra).to(u.radian).value
#     dec = uvw_frame.phase.dec.to(u.radian).value
#     lonrad = lon.to(u.radian).value - ha
#     latrad = dec  # lat.to(u.radian).value +
#     sinlat = np.sin(latrad)
#     coslat = np.cos(latrad)
#     sinlon = np.sin(lonrad)
#     coslon = np.cos(lonrad)
#     north = [-sinlat * coslon,
#              -sinlat * sinlon,
#              coslat]
#     east = [-sinlon, coslon, 0]
#     up = [coslat * coslon, coslat * sinlon, sinlat]
#     R = np.array([east, north, up])
#
#     if is_unitspherical:
#         # don't need to do distance calculation
#         p = itrs_coo.cartesian.xyz.value
#         diff = p
#         penu = R.dot(diff)
#
#         rep = CartesianRepresentation(x=u.Quantity(penu[0], u.one, copy=False),
#                                       y=u.Quantity(penu[1], u.one, copy=False),
#                                       z=u.Quantity(penu[2], u.one, copy=False),
#                                       copy=False)
#     else:
#         p = itrs_coo.cartesian.xyz
#         p0 = ITRS(*uvw_frame.location.geocentric, obstime=uvw_frame.obstime).cartesian.xyz
#         diff = (p.T - p0).T
#         penu = R.dot(diff)
#
#         rep = CartesianRepresentation(x=penu[0],  # u.Quantity(penu[0],u.m,copy=False),
#                                       y=penu[1],  # u.Quantity(penu[1],u.m,copy=False),
#                                       z=penu[2],  # u.Quantity(penu[2],u.m,copy=False),
#                                       copy=False)
#
#     return uvw_frame.realize_frame(rep)
#
#
# @frame_transform_graph.transform(FunctionTransform, UVW, ITRS)
# def uvw_to_itrs(uvw_coo, itrs_frame):
#     # p = itrs_frame.cartesian.xyz.to(u.m).value
#     # p0 = np.array(enu_coo.location.to(u.m).value)
#     # p = np.array(itrs_frame.location.to(u.m).value)
#
#     lon, lat, height = uvw_coo.location.to_geodetic('WGS84')
#     lst_ = AltAz(alt=90 * u.deg, az=0 * u.deg, location=uvw_coo.location, obstime=uvw_coo.obstime).transform_to(ICRS).ra
#     lst = uvw_coo.obstime.sidereal_time('apparent', lon)
#     # print(lst.deg,lst_.deg)
#
#     ha = (lst - uvw_coo.phase.ra).to(u.radian).value
#     dec = uvw_coo.phase.dec.to(u.radian).value
#     lonrad = lon.to(u.radian).value - ha
#     latrad = dec  # lat.to(u.radian).value +
#     sinlat = np.sin(latrad)
#     coslat = np.cos(latrad)
#     sinlon = np.sin(lonrad)
#     coslon = np.cos(lonrad)
#
#     north = [-sinlat * coslon,
#              -sinlat * sinlon,
#              coslat]
#     east = [-sinlon, coslon, 0]
#     up = [coslat * coslon, coslat * sinlon, sinlat]
#     R = np.array([east, north, up])
#
#     if isinstance(uvw_coo.data, UnitSphericalRepresentation) or uvw_coo.cartesian.x.unit == u.one:
#         diff = R.T.dot(uvw_coo.cartesian.xyz)
#         p = diff
#         rep = CartesianRepresentation(x=u.Quantity(p[0], u.one, copy=False),
#                                       y=u.Quantity(p[1], u.one, copy=False),
#                                       z=u.Quantity(p[2], u.one, copy=False),
#                                       copy=False)
#     else:
#         diff = R.T.dot(uvw_coo.cartesian.xyz)
#         p0 = ITRS(*uvw_coo.location.geocentric, obstime=uvw_coo.obstime).cartesian.xyz
#         # print (R,diff)
#         p = (diff.T + p0).T
#         # print (p)
#         rep = CartesianRepresentation(x=p[0],  # u.Quantity(p[0],u.m,copy=False),
#                                       y=p[1],  # u.Quantity(p[1],u.m,copy=False),
#                                       z=p[2],  # u.Quantity(p[2],u.m,copy=False),
#                                       copy=False)
#
#     return itrs_frame.realize_frame(rep)
#
#     # return ITRS(*p*u.m,obstime=enu_coo.obstime).transform_to(itrs_frame)
#
#
# @frame_transform_graph.transform(FunctionTransform, UVW, UVW)
# def uvw_to_uvw(from_coo, to_frame):
#     # for now we just implement this through ITRS to make sure we get everything
#     # covered
#     return from_coo.transform_to(ITRS(obstime=from_coo.obstime)).transform_to(to_frame)