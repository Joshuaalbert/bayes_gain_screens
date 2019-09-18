
from .common_setup import *
from ..misc import make_example_datapack
import numpy as np
import pylab as plt
import os

from bayes_filter.plotting import animate_datapack, plot_vornoi_map



def test_plotdatapack():
    # dp = DatapackPlotter(datapack='/home/albert/git/bayes_tec/scripts/data/P126+65_compact_full_raw.h5')
    # dp.plot(ant_sel=None, pol_sel=slice(0,1,1), time_sel=slice(0,1,1), fignames=['test_fig.png'], solset='sol000', observable='phase', plot_facet_idx=True,
    #         labels_in_radec=True, show=False)

    datapack = os.path.join(TEST_FOLDER,'plotting_test.h5')

    datapack = make_example_datapack(35, 2, 1, pols=['XX'], index_n=1,obs_type='DDTEC',
                                     clobber=True, name=os.path.join(TEST_FOLDER,'plotting_test.h5'))

    animate_datapack(datapack,
           os.path.join(TEST_FOLDER, 'test_plotting_tecphase'),num_processes=1,observable='tec',labels_in_radec=True,
                    solset='sol000', plot_facet_idx=True, tec_eval_freq=140e6,phase_wrap=True)

    animate_datapack(datapack,
                     os.path.join(TEST_FOLDER, 'test_plotting_tec'), num_processes=1, observable='tec',
                     labels_in_radec=True,
                     solset='sol000', plot_facet_idx=True,phase_wrap=False)

def test_plot_vornoi_map():
    fig, ax = plt.subplots(1,1,figsize=(5,5))
    points = np.random.uniform(size=(5,2))
    colors = np.random.uniform(size=5)
    plot_vornoi_map(points, colors, ax=ax, alpha=1.,radius=100,relim=True)
    #plt.show()
    plt.close('all')