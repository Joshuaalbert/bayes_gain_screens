"""
This will deploy the probabilistic screen solver on an H5Parm
"""

from bayes_gain_screens.deploy import Deployment
from bayes_gain_screens.datapack import DataPack
from bayes_gain_screens.misc import make_soltab
import numpy as np
import argparse
import os
from bayes_gain_screens import logging


def add_args(parser):
    def dim_selection(s: str):
        if s.lower() == 'none':
            return None
        if '/' in s:  # slice
            s = s.split("/")
            if len(s) != 3:
                raise ValueError("Proper slice notations is 'start/stop/step'")
            return slice(int(s[0]) if s[0].lower() != 'none' else None,
                         int(s[1]) if s[1].lower() != 'none' else None,
                         int(s[2]) if s[2].lower() != 'none' else None)
        if ',' in s:
            s = s.replace(']', "").replace('[', '')
            s = s.split(',')
            s = [int(v.strip()) for v in s]
            return s
        return s

    def model_parse(s: str):
        s = s.replace(']', "").replace('[', '')
        s = s.split(',')
        s = [v.upper().strip() for v in s]
        return s

    def obstype_parse(s: str):
        s = s.upper()
        if s not in ['TEC', 'DTEC', 'DDTEC']:
            raise ValueError("{} not a valid obstype.".format(s))
        return s

    def list_parse(s: str):
        s = s.replace('[','')
        s = s.replace(']','')
        return [int(d.strip()) for d in s.split(',')]

    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.register("type", "axes_selection", dim_selection)
    parser.register("type", "model_selection", model_parse)
    parser.register("type", "obstype_type", obstype_parse)
    parser.register("type", "list", list_parse)

    optional = parser._action_groups.pop()  # Edited this line
    parser._action_groups.append(optional)  # added this line
    required = parser.add_argument_group('Required arguments')


    required.add_argument("--datapack", type=str,
                          default=None,
                          help="""Datapack input, a losoto h5parm.""", required=True)
    required.add_argument("--ref_dir", type=int,
                          default=None,
                          help="""Reference direction, should be a bright direction.""", required=True)
    optional.add_argument("--tec_solset", type=str,
                          default='sol000',
                          help="""solset to get tec000 from for solve.""")
    optional.add_argument("--phase_solset", type=str,
                          default='sol000',
                          help="""solset to get phase000 from for phase referencing.""")

    # optional arguments
    optional.add_argument("--deployment_type", type=str,
                          default='directional',
                          help="""Type of solve: ['directional','non_integral', 'tomographic'].""", required=False)
    optional.add_argument("--use_vec_kernels", type="bool",
                          default=False,
                          help="""Whether to include directional kernels with vectorised amps.""", required=False)
    optional.add_argument("--flag_directions", type="list",
                          default=None,
                          help="""Flag specific directions for sure. In addition to regular filtering.""", required=False)
    optional.add_argument("--ant", type="axes_selection", default=None,
                          help="""The antennas selection: None, regex RS*, or slice format <start>/<stop>/<step>.\n""")
    optional.add_argument("--time", type="axes_selection", default=None,
                          help="""The antennas selection: None, or slice format <start>/<stop>/<step>.\n""")
    optional.add_argument("--dir", type="axes_selection", default=None,
                          help="""The direction selection: None, regex patch_???, or slice format <start>/<stop>/<step>.\n""")
    optional.add_argument("--pol", type="axes_selection", default=slice(0, 1, 1),
                          help="""The polarization selection: None, list XX,XY,YX,YY, regex X?, or slice format <start>/<stop>/<step>.\n""")
    optional.add_argument("--freq", type="axes_selection", default=None,
                          help="""The channel selection: None, or slice format <start>/<stop>/<step>.\n""")
    optional.add_argument("--output_folder", type=str, default='./deployment_directional',
                          help="Folder to store output.")
    optional.add_argument("--srl_file", type=str, default=None,
                          help="SRL file containing list of sources in field. Used to choose screen points.")
    optional.add_argument("--min_spacing_arcmin", type=float, default=1.,
                          help="Minimum angular distance between screen points.")
    optional.add_argument("--max_N", type=int, default=250,
                          help="Maximum number of screen points.")
    optional.add_argument("--flux_limit", type=float, default=0.05,
                          help="Minimum brightness to be considered for a screen point [Jy/beam].")
    optional.add_argument("--block_size", type=int, default=10,
                          help="Number of timesteps to solve independently at once (more boosts signal to noise, but the ionosphere might change).")

def run_paper3():
    datapack = DataPack('/net/lofar1/data1/albert/imaging/data/P126+65_compact_raw/P126+65_full_compact_raw_v9.h5', readonly=False)
    make_soltab(datapack, from_solset='sol000', from_soltab='phase000', to_solset='sol000', to_soltab='tec000')
    datapack.current_solset = 'sol000'
    reinout_flags = np.load('/home/albert/lofar1_1/imaging/data/flagsTECBay.npy')[None, ...]
    tec_uncert = np.where(reinout_flags == 1., np.inf, 1.)  # uncertainty in mTECU
    datapack.weights_tec = tec_uncert
    reinout_tec = np.load('/net/rijn/data2/rvweeren/P126+65_recall/ClockTEC/TECBay.npy')[None, ...] * 1e3
    reinout_tec[:, 14, ...] = 0.
    datapack.tec = reinout_tec

    main('directional', datapack, ref_dir=14, output_folder='paper3_directional_deployment', min_spacing_arcmin=1., max_N=250,
         flux_limit=0.05, block_size=10, srl_file='/home/albert/ftp/image.pybdsm.srl.fits', ant=None, time=None,
         dir=None, pol=slice(0, 1, 1), freq=None)

def main(deployment_type, datapack, tec_solset, phase_solset, ref_dir, output_folder, min_spacing_arcmin, max_N, flux_limit, block_size, srl_file, use_vec_kernels, ant, time, dir, pol, freq):
    if deployment_type not in ['directional','non_integral', 'tomographic']:
        raise ValueError("Invalid deployment_type".format(deployment_type))
    if deployment_type == 'directional':
        from bayes_gain_screens.directional_models import generate_models
        directional_deploy = True
    if deployment_type == 'non_integral':
        from bayes_gain_screens.non_integral_models import generate_models
        directional_deploy = False
    if deployment_type == 'tomographic':
        from bayes_gain_screens.tomographic_models import generate_models
        directional_deploy = False

    deployment = Deployment(datapack,
                            ref_dir_idx=ref_dir,
                            tec_solset=tec_solset,
                            phase_solset=phase_solset,
                            flux_limit=flux_limit,
                            max_N=max_N,
                            min_spacing_arcmin=min_spacing_arcmin,
                            srl_file=srl_file,
                            ant=ant,
                            dir=dir,
                            time=time,
                            freq=freq,
                            pol=pol,
                            directional_deploy=directional_deploy,
                            block_size=block_size,
                            working_dir=os.path.abspath(output_folder))
    deployment.run(generate_models, use_vec_kernels=use_vec_kernels)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_args(parser)
    flags, unparsed = parser.parse_known_args()
    logging.info("Running with:")
    for option, value in vars(flags).items():
        logging.info("    {} -> {}".format(option, value))
    main(**vars(flags))
