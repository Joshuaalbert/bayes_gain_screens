from .common_setup import *

from ..datapack import DataPack
from ..misc import make_example_datapack
import os
import numpy as np

def test_datapack():
    datapack = make_example_datapack(4,2,1,["X"],obs_type='DTEC',clobber=True, name=os.path.join(TEST_FOLDER,'test_datapack_data.h5'))
    phase,axes = datapack.phase
    datapack.phase = phase+1.
    phasep1, axes = datapack.phase
    assert np.all(np.isclose(phasep1, phase+1.))
    datapack.select(ant='RS509', time=slice(0,1,1))
    phase,axes = datapack.phase
    print(phase.shape)
    assert np.all(phase == phasep1[:,:,-1:,:,:1])
    datapack.select(ant='RS*', time=slice(0, 1, 1))
    phase, axes = datapack.phase
    for  a in axes['ant']:
        assert b'RS' in a
    assert len(axes['ant']) == 14
    dtec = np.mean(phase,axis=-2)
    datapack.select(ant='RS*', time=slice(0,1,1),freq=slice(0,1,1))
    datapack.tec = dtec

    array_to_set = np.ones((1, 2, 62, 2, 1))
    datapack.select(dir=slice(0,2,1), ant=slice(0,62,1), time=slice(0,1,1))
    datapack.phase = array_to_set


# def test_datapack():
#     datapack = DataPack('test.h5',readonly=False)
#     datapack.current_solset = 'sol000'
#     datapack.select(ant="RS*")
#     phase, axes = datapack.phase
#     print(axes)
#     datapack.phase = np.ones_like(phase)
#     print(datapack.phase)
#     # # datapack.add_solset('test')
#     # print(datapack.soltabs)
#     # # print(datapack.directions)
#     # datapack.add_soltab('foo', ant=['CS001HBA0'], dir=['patch_0'], freq=[0., 1.])
#     # print(datapack)
#     # datapack.delete_soltab('foo')