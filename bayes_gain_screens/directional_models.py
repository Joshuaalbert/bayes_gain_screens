from gpflow import settings
import numpy as np

from .directional_kernels import (GreatCircleRBF, GreatCircleM32, GreatCircleM52, GreatCircleM12, \
    GreatCircleRQ, VectorAmplitudeWrapper, DirectionalKernel, ThinLayerM12, ThinLayerM32, ThinLayerM52, \
    ThinLayerRQ, ThinLayerRBF)
from .model import HGPR
from gpflow.kernels import Matern52, Matern32, Matern12, RBF, ArcCosine
from . import logging

def gpflow_kernel(kernel, dims=3, **kwargs):
    kern_map = dict(RBF=RBF, M32=Matern32, M52=Matern52, M12=Matern12,
                    ArcCosine=ArcCosine,
                    GreatCircleRBF=GreatCircleRBF, GreatCircleRQ=GreatCircleRQ,
                    GreatCircleM12=GreatCircleM12, GreatCircleM32=GreatCircleM32,
                    GreatCircleM52=GreatCircleM52, ThinLayerRBF=ThinLayerRBF, ThinLayerM52=ThinLayerM52,
                    ThinLayerM32=ThinLayerM32, ThinLayerM12=ThinLayerM12, ThinLayerRQ=ThinLayerRQ)
    kern = kern_map.get(kernel, None)
    if kern is None:
        raise ValueError("{} not valid kernel".format(kernel))
    return kern(dims, **kwargs)


def generate_models(X, Y, Y_var, ref_direction, reg_param=1., parallel_iterations=10, anisotropic=False,
                    use_vec_kernels=False, **kwargs):
    logging.info("Generating directional GP models.")
    amplitude = None
    if len(Y.shape) == 3:
        Y_flag = np.copy(Y)
        for _ in range(3):
            # B, T, N -> B*N, T -> T
            amplitude = np.nanstd(Y_flag.transpose((0,2,1)).reshape((-1, Y.shape[1])), axis=0)
            Y_flag[Y_flag > 3.*amplitude[None,:, None]] = np.nan
        amplitude = np.nanstd(Y_flag.transpose((0,2,1)).reshape((-1, Y.shape[1])), axis=0)
        amplitude = np.where(np.isnan(amplitude), 1., amplitude)

    # h*l = hpd -> l = hpd / h
    dir_kernels = [
        gpflow_kernel('GreatCircleRBF', dims=3, variance=1. ** 2, hpd=1.*np.pi/180.),
        gpflow_kernel('GreatCircleM52', dims=3, variance=1. ** 2, hpd=1.*np.pi/180.),
        gpflow_kernel('GreatCircleM32', dims=3, variance=1. ** 2, hpd=1.*np.pi/180.),
        gpflow_kernel('GreatCircleM12', dims=3, variance=1. ** 2, hpd=1.*np.pi/180.),
        gpflow_kernel('GreatCircleRQ', dims=3, variance=1. ** 2, hpd=1.*np.pi/180., alpha=10.),
        gpflow_kernel('ThinLayerRBF', dims=3, variance=1. ** 2, hpd=15., height=200.),
        gpflow_kernel('ThinLayerM52', dims=3, variance=1. ** 2, hpd=15., height=200.),
        gpflow_kernel('ThinLayerM32', dims=3, variance=1. ** 2, hpd=15., height=200.),
        gpflow_kernel('ThinLayerM12', dims=3, variance=1. ** 2, hpd=15., height=200.),
        gpflow_kernel('ThinLayerRQ', dims=3, variance=1. ** 2, hpd=15., alpha=10., height=200.),
    ]

    kernels = []
    for d in dir_kernels:
        kernels.append(DirectionalKernel(ref_direction=ref_direction,
                                         anisotropic=False,
                                         inner_kernel=d,
                                         amplitude=amplitude,
                                         obs_type='DDTEC'))


    if use_vec_kernels:
        dir_kernels = [
            gpflow_kernel('GreatCircleRBF', dims=3, variance=1. ** 2, hpd=1. * np.pi / 180.),
            gpflow_kernel('GreatCircleM52', dims=3, variance=1. ** 2, hpd=1. * np.pi / 180.),
            gpflow_kernel('GreatCircleM32', dims=3, variance=1. ** 2, hpd=1. * np.pi / 180.),
            gpflow_kernel('GreatCircleM12', dims=3, variance=1. ** 2, hpd=1. * np.pi / 180.),
            gpflow_kernel('GreatCircleRQ', dims=3, variance=1. ** 2, hpd=1. * np.pi / 180., alpha=10.),
            gpflow_kernel('ThinLayerRBF', dims=3, variance=1. ** 2, hpd=15., height=200.),
            gpflow_kernel('ThinLayerM52', dims=3, variance=1. ** 2, hpd=15., height=200.),
            gpflow_kernel('ThinLayerM32', dims=3, variance=1. ** 2, hpd=15., height=200.),
            gpflow_kernel('ThinLayerM12', dims=3, variance=1. ** 2, hpd=15., height=200.),
            gpflow_kernel('ThinLayerRQ', dims=3, variance=1. ** 2, hpd=15., alpha=10., height=200.),
        ]

        for d in dir_kernels:
            kernels.append(VectorAmplitudeWrapper(
                inner_kernel=d,
                amplitude=amplitude))

    models = [HGPR(X, Y, Y_var, kern, regularisation_param=reg_param, parallel_iterations=parallel_iterations,
                   name="HGPR_{}".format(kern.name))
              for kern in kernels]

    return models
