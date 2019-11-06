from .tomographic_kernel import TomographicKernel
from .model import HGPR
from .directional_models import gpflow_kernel


def generate_models(X, Y, Y_var, ref_direction, ref_location, reg_param=1., parallel_iterations=10, **kwargs):
    fed_settings = [('RQ', dict(variance=9., lengthscales=15., alpha=10.))
                    ('RBF', dict(variance=9., lengthscales=15.)),
                    ('M52', dict(variance=9., lengthscales=15.)),
                    ('M32', dict(variance=9., lengthscales=15.)),
                    ('M12', dict(variance=9., lengthscales=15.))]
    kernels = []
    for k in fed_settings:
        kernels.append(TomographicKernel(a=250., b=100.,
                                         ref_direction=ref_direction,
                                         ref_location=ref_location,
                                         anisotropic=False,
                                         fed_kernel=gpflow_kernel(k[0], **k[1]),
                                         obs_type='DDTEC',
                                         resolution=5,
                                         ionosphere_type='flat'))

    models = [HGPR(X, Y, Y_var, kern, regularisation_param=reg_param, parallel_iterations=parallel_iterations,
                   name='HGPR_{}'.format(kern.name))
              for kern in kernels]

    return models
