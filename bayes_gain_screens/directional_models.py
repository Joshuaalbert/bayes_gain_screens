from gpflow import settings
import numpy as np

from .directional_kernels import (GreatCircleRBF, GreatCircleM32, GreatCircleM52, GreatCircleM12, \
                                  GreatCircleRQ, ThinLayerM12, ThinLayerM32,
                                  ThinLayerM52, \
                                  ThinLayerRQ, ThinLayerRBF, DirectionalKernelThinLayerFull, fix_kernel, VecKernel)
from .model import HGPR
from gpflow.kernels import Matern52, Matern32, Matern12, RBF, ArcCosine, RationalQuadratic
from . import logging


def gpflow_kernel(kernel, dims=3, **kwargs):
    kern_map = dict(RBF=fix_kernel(RBF), M32=fix_kernel(Matern32), M52=fix_kernel(Matern52), M12=fix_kernel(Matern12),
                    RQ=fix_kernel(RationalQuadratic),
                    ArcCosine=ArcCosine,
                    GreatCircleRBF=GreatCircleRBF, GreatCircleRQ=GreatCircleRQ,
                    GreatCircleM12=GreatCircleM12, GreatCircleM32=GreatCircleM32,
                    GreatCircleM52=GreatCircleM52, ThinLayerRBF=ThinLayerRBF, ThinLayerM52=ThinLayerM52,
                    ThinLayerM32=ThinLayerM32, ThinLayerM12=ThinLayerM12, ThinLayerRQ=ThinLayerRQ)
    kern = kern_map.get(kernel, None)
    if kern is None:
        raise ValueError("{} not valid kernel".format(kernel))
    return kern(dims, **kwargs)


def generate_models(X, Y, Y_var, ref_location=None, ref_direction=None,
                    **kwargs):
    """
    Generates hypothesis kernels

    :param X: np.array [Na, Nd, 6]
    :param Y: np.array [B, Na, Nd]
    :param Y_var: np.array [N, Na, Nd]
    :param ref_location: [3]
    :param ref_direction: [3]
    :param kwargs:
    :return:
    """
    logging.info("Generating directional GP models.")
    amplitude = None
    if len(Y.shape) == 3:
        # logging.info(Y.shape)
        logging.info(np.where(Y_var == np.inf))
        amplitude = np.sqrt(np.nanmean(np.nanmean( np.where(np.isinf(Y_var),np.nan, Y**2), axis=0), axis=-1))
        amplitude = np.where(np.isnan(amplitude), 1., amplitude)

    # h*l = hpd -> l = hpd / h
    inner_kernels = [
        gpflow_kernel('RBF', dims=3, variance=1. ** 2, lengthscales=15.),
        gpflow_kernel('M52', dims=3, variance=1. ** 2, lengthscales=15.),
        gpflow_kernel('M32', dims=3, variance=1. ** 2, lengthscales=15.),
        gpflow_kernel('M12', dims=3, variance=1. ** 2, lengthscales=15.),
        gpflow_kernel('RQ', dims=3, variance=1. ** 2, lengthscales=15., alpha=10.),
    ]

    kernels = []
    for d in inner_kernels:
        kernels.append(DirectionalKernelThinLayerFull(ref_direction=ref_direction,
                                                      ref_location=ref_location,
                                                      a=250.,
                                                      inner_kernel=d,
                                                      amplitude=amplitude,
                                                      obs_type='DDTEC'))
    inner_kernels = [
        gpflow_kernel('RBF', dims=3, variance=1. ** 2, lengthscales=1.*np.pi/180.),
        gpflow_kernel('M52', dims=3, variance=1. ** 2, lengthscales=1.*np.pi/180.),
        gpflow_kernel('M32', dims=3, variance=1. ** 2, lengthscales=1.*np.pi/180.),
        gpflow_kernel('M12', dims=3, variance=1. ** 2, lengthscales=1.*np.pi/180.),
        gpflow_kernel('RQ', dims=3, variance=1. ** 2, lengthscales=1.*np.pi/180., alpha=10.),
    ]

    for d in inner_kernels:
        kernels.append(VecKernel(
                                  inner_kernel=d,
                                  amplitude=amplitude))

    models = [HGPR(X, Y, Y_var, kern, caption="HGPR_{}".format(kern.caption)) for kern in kernels]

    return models
