from bayes_filter.deploy import Deployment
from bayes_filter.tomographic_models import generate_models
from bayes_filter.datapack import DataPack
from bayes_filter.misc import make_soltab
import numpy as np


if __name__ == '__main__':
    datapack = DataPack('/home/albert/lofar1_1/imaging/data/P126+65_compact_raw/P126+65_full_compact_raw_v11.h5', readonly=False)
    make_soltab(datapack, from_solset='sol000', from_soltab='phase000', to_solset='sol000', to_soltab='tec000')
    datapack.current_solset = 'sol000'
    reinout_flags = np.load('/home/albert/lofar1_1/imaging/data/flagsTECBay.npy')[None, ...]
    tec_uncert = np.where(reinout_flags == 1., np.inf, 1.)  # uncertainty in mTECU
    datapack.weights_tec = tec_uncert
    reinout_tec = np.load('/net/rijn/data2/rvweeren/P126+65_recall/ClockTEC/TECBay.npy')[None, ...] * 1e3
    reinout_tec[:, 14, ...] = 0.
    datapack.tec = reinout_tec
    deployment = Deployment(datapack,
                            ref_dir_idx=14,
                            solset='sol000',
                            flux_limit=0.05,
                            max_N=250,
                            min_spacing_arcmin=1.,
                            srl_file = '/home/albert/ftp/image.pybdsm.srl.fits',
                            ant = slice(0, None,1),
                            dir = None,
                            time = slice(0, None, 1),
                            freq = None,
                            pol = slice(0, 1, 1),
                            directional_deploy = False,
                            block_size = 40,
                            working_dir = './deployment_tomographic')
    deployment.run(generate_models)


