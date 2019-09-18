import astropy.coordinates as ac
import astropy.time as at
from .frames import ENU
from .settings import dist_type, angle_type
import numpy as np
import tensorflow as tf
from .callbacks import Callback
from .settings import float_type



def tf_coord_transform(transform):
    def tf_transform(X):
        return tf.py_function(lambda X: transform(X.numpy()), [X], X.dtype)
    return tf_transform

#TODO: make a callback
def itrs_to_enu_6D(X, ref_location=None):
    """
    Convert the given coordinates from ITRS to ENU

    :param X: float array [b0,...,bB,6]
        The coordinates are ordered [time, ra, dec, itrs.x, itrs.y, itrs.z]
    :param ref_location: float array [3]
        Point about which to rotate frame.
    :return: float array [b0,...,bB, 7]
        The transforms coordinates.
    """
    time = np.unique(X[..., 0])
    if time.size > 1:
        raise ValueError("Times should be the same.")

    shape = X.shape[:-1]
    X = X.reshape((-1, 6))
    if ref_location is None:
        ref_location = X[0,3:]
    obstime = at.Time(time / 86400., format='mjd')
    location = ac.ITRS(x=ref_location[0] * dist_type, y=ref_location[1] * dist_type, z=ref_location[2] * dist_type)
    enu = ENU(location=location, obstime=obstime)
    antennas = ac.ITRS(x=X[:, 3] * dist_type, y=X[:, 4] * dist_type, z=X[:, 5] * dist_type, obstime=obstime)
    antennas = antennas.transform_to(enu).cartesian.xyz.to(dist_type).value.T
    directions = ac.ICRS(ra=X[:, 1] * angle_type, dec=X[:, 2] * angle_type)
    directions = directions.transform_to(enu).cartesian.xyz.value.T
    return np.concatenate([X[:,0:1], directions, antennas], axis=1).reshape(shape+(7,))

#TODO: make a callback

def itrs_to_enu_with_references(ref_antenna=None, ref_direction=None, ref_location=None):
    """
    Wrapper that creates the function to convert the given coordinates from ITRS to ENU

    :param ref_antenna: float array [3]
        Location of reference antenna in ITRS.
    :param ref_direction: float array [2]
        RA and DEC of reference direction.
    :param ref_location: float array [3]
        Point about which to rotate frame.
    """

    def transform(X, ref_location=ref_location):
        """
        Convert the given coordinates from ITRS to ENU

        :param X: float array [Nd, Na,6]
            The coordinates are ordered [time, ra, dec, itrs.x, itrs.y, itrs.z]
        :return: float array [Nd, Na, 7(10(13))]
            The transforms coordinates.
        """
        time = np.unique(X[..., 0])
        if time.size > 1:
            raise ValueError("Times should be the same.")
        shape = X.shape[:-1]
        X = X.reshape((-1, 6))
        if ref_location is None:
            ref_location = X[0,3:6]
        obstime = at.Time(time / 86400., format='mjd')
        location = ac.ITRS(x=ref_location[0] * dist_type, y=ref_location[1] * dist_type, z=ref_location[2] * dist_type)
        ref_ant = ac.ITRS(x=ref_antenna[0] * dist_type, y=ref_antenna[1] * dist_type, z=ref_antenna[2] * dist_type, obstime=obstime)
        ref_dir = ac.ICRS(ra=ref_direction[0] * angle_type, dec=ref_direction[1]  * angle_type)
        enu = ENU(location=location, obstime=obstime)
        ref_ant = ref_ant.transform_to(enu).cartesian.xyz.to(dist_type).value.T
        ref_dir = ref_dir.transform_to(enu).cartesian.xyz.value.T
        antennas = ac.ITRS(x=X[:, 3] * dist_type, y=X[:, 4] * dist_type, z=X[:, 5] * dist_type, obstime=obstime)
        antennas = antennas.transform_to(enu).cartesian.xyz.to(dist_type).value.T
        directions = ac.ICRS(ra=X[:, 1] * angle_type, dec=X[:, 2] * angle_type)
        directions = directions.transform_to(enu).cartesian.xyz.value.T
        result = np.concatenate([X[:,0:1], directions, antennas], axis=1)#
        if ref_antenna is not None:
            result = np.concatenate([result, np.tile(ref_ant, (result.shape[0], 1))], axis=-1)
        if ref_direction is not None:
            result = np.concatenate([result, np.tile(ref_dir, (result.shape[0], 1))],axis=-1)
        result = result.reshape(shape+result.shape[-1:])
        return result
    return transform

class ITRSToENUWithReferences(Callback):
    def __init__(self,ref_antenna=None, ref_direction=None, ref_location=None):
        """
        Wrapper that creates the function to convert the given coordinates from ITRS to ENU

        :param ref_antenna: float array [3]
            Location of reference antenna in ITRS.
        :param ref_direction: float array [2]
            RA and DEC of reference direction.
        :param ref_location: float array [3]
            Point about which to rotate frame.
        """
        super(ITRSToENUWithReferences, self).__init__(ref_antenna=ref_antenna,
                                                      ref_direction=ref_direction,
                                                      ref_location=ref_location)

    def generate(self, ref_antenna, ref_direction, ref_location):
        self.output_dtypes = [float_type]
        self.name = 'ITRSToENUWithReferences'
        self.squeeze = True

        def transform(X,ref_location=ref_location):
            """
            Convert the given coordinates from ITRS to ENU

            :param X: float array [Nd, Na,6]
                The coordinates are ordered [time, ra, dec, itrs.x, itrs.y, itrs.z]
            :return: float array [Nd, Na, 7(10(13))]
                The transforms coordinates.
            """
            time = np.unique(X[..., 0])
            if time.size > 1:
                raise ValueError("Times should be the same.")
            shape = X.shape[:-1]
            X = X.reshape((-1, 6))
            if ref_location is None:
                ref_location = X[0,3:6]
            obstime = at.Time(time / 86400., format='mjd')
            location = ac.ITRS(x=ref_location[0] * dist_type, y=ref_location[1] * dist_type, z=ref_location[2] * dist_type)
            ref_ant = ac.ITRS(x=ref_antenna[0] * dist_type, y=ref_antenna[1] * dist_type, z=ref_antenna[2] * dist_type, obstime=obstime)
            ref_dir = ac.ICRS(ra=ref_direction[0] * angle_type, dec=ref_direction[1]  * angle_type)
            enu = ENU(location=location, obstime=obstime)
            ref_ant = ref_ant.transform_to(enu).cartesian.xyz.to(dist_type).value.T
            ref_dir = ref_dir.transform_to(enu).cartesian.xyz.value.T
            antennas = ac.ITRS(x=X[:, 3] * dist_type, y=X[:, 4] * dist_type, z=X[:, 5] * dist_type, obstime=obstime)
            antennas = antennas.transform_to(enu).cartesian.xyz.to(dist_type).value.T
            directions = ac.ICRS(ra=X[:, 1] * angle_type, dec=X[:, 2] * angle_type)
            directions = directions.transform_to(enu).cartesian.xyz.value.T
            result = np.concatenate([X[:,0:1], directions, antennas], axis=1)#
            if ref_antenna is not None:
                result = np.concatenate([result, np.tile(ref_ant, (result.shape[0], 1))], axis=-1)
            if ref_direction is not None:
                result = np.concatenate([result, np.tile(ref_dir, (result.shape[0], 1))],axis=-1)
            result = result.reshape(shape+result.shape[-1:])
            return result
        return transform

class ITRSToENUWithReferences_v2(Callback):
    def __init__(self,ref_antenna=None, ref_direction=None, ref_location=None):
        """
        Wrapper that creates the function to convert the given coordinates from ITRS to ENU

        :param ref_antenna: float array [3]
            Location of reference antenna in ITRS.
        :param ref_direction: float array [2]
            RA and DEC of reference direction.
        :param ref_location: float array [3]
            Point about which to rotate frame.
        """
        super(ITRSToENUWithReferences_v2, self).__init__(ref_antenna=ref_antenna,
                                                      ref_direction=ref_direction,
                                                      ref_location=ref_location)
    def generate(self, ref_antenna, ref_direction, ref_location):
        self.output_dtypes = [float_type, float_type, float_type]
        self.name = 'ITRSToENUWithReferences_v2'
        self.squeeze = True

        from .misc import make_coord_array

        def transform(Xt, Xd, Xa):
            """
            Convert the given coordinates from ITRS to ENU

            :param X: float array [Nd, Na,6]
                The coordinates are ordered [time, ra, dec, itrs.x, itrs.y, itrs.z]
            :return: float array [Nt, Nd, Na, 6] [Nt, 3] [Nt, 3]
                The transforms coordinates, ref_ants, ref_dirs
            """
            X_out = []
            ref_ant_out = []
            ref_dir_out = []
            for t in range(Xt.shape[0]):
                obstime = at.Time(Xt[t, 0] / 86400., format='mjd')
                location = ac.ITRS(x=ref_location[0] * dist_type, y=ref_location[1] * dist_type, z=ref_location[2] * dist_type)
                ref_ant = ac.ITRS(x=ref_antenna[0] * dist_type, y=ref_antenna[1] * dist_type, z=ref_antenna[2] * dist_type, obstime=obstime)
                ref_dir = ac.ICRS(ra=ref_direction[0] * angle_type, dec=ref_direction[1]  * angle_type)
                enu = ENU(location=location, obstime=obstime)
                ref_ant = ref_ant.transform_to(enu).cartesian.xyz.to(dist_type).value.T
                ref_dir = ref_dir.transform_to(enu).cartesian.xyz.value.T
                antennas = ac.ITRS(x=Xa[:, 0] * dist_type, y=Xa[:, 1] * dist_type, z=Xa[:, 2] * dist_type, obstime=obstime)
                antennas = antennas.transform_to(enu).cartesian.xyz.to(dist_type).value.T
                directions = ac.ICRS(ra=Xd[:, 0] * angle_type, dec=Xd[:, 1] * angle_type)
                directions = directions.transform_to(enu).cartesian.xyz.value.T
                X_out.append(make_coord_array(directions, antennas, flat=False))
                ref_ant_out.append(ref_ant)
                ref_dir_out.append(ref_dir)
            X = np.stack(X_out, axis=0)
            ref_ant = np.concatenate(ref_ant_out,axis=0)
            ref_dir = np.concatenate(ref_dir_out,axis=0)
            return [X, ref_ant, ref_dir]
        return transform