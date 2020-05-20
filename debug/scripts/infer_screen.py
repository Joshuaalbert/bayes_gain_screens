"""
This will deploy the probabilistic screen solver on an H5Parm
"""

from bayes_gain_screens.deploy import Deployment
from bayes_gain_screens.plotting import animate_datapack
import argparse
import os

def link_overwrite(src, dst):
    if os.path.islink(dst):
        print("Unlinking pre-existing sym link {}".format(dst))
        os.unlink(dst)
    print("Linking {} -> {}".format(src, dst))
    os.symlink(src, dst)

def main(data_dir, working_dir, obs_num, ref_dir, deployment_type, block_size, ref_image_fits, ncpu, max_N):
    generate_models = None
    if deployment_type not in ['directional', 'non_integral', 'tomographic']:
        raise ValueError("Invalid deployment_type".format(deployment_type))
    if deployment_type == 'directional':
        from bayes_gain_screens.directional_models import generate_models
    if deployment_type == 'non_integral':
        from bayes_gain_screens.non_integral_models import generate_models
    if deployment_type == 'tomographic':
        from bayes_gain_screens.tomographic_models import generate_models

    dds5_h5parm = os.path.join(data_dir, 'L{}_{}_merged.h5'.format(obs_num, 'DDS5_full'))
    dds6_h5parm = os.path.join(working_dir, 'L{}_{}_merged.h5'.format(obs_num, 'DDS6_full'))
    linked_dds6_h5parm = os.path.join(data_dir, 'L{}_{}_merged.h5'.format(obs_num, 'DDS6_full'))

    deployment = Deployment(dds5_h5parm,
                            dds6_h5parm,
                            ref_dir_idx=ref_dir,
                            tec_solset='directionally_referenced',
                            phase_solset='smoothed000',
                            flux_limit=0.01,
                            max_N=max_N,
                            min_spacing_arcmin=4.,
                            ref_image_fits=ref_image_fits,
                            ant=None,
                            dir=None,
                            time=None,
                            freq=None,
                            pol=slice(0, 1, 1),
                            block_size=block_size,
                            working_dir=working_dir,
                            remake_posterior_solsets=True)

    link_overwrite(dds6_h5parm, linked_dds6_h5parm)

    deployment.run(generate_models)

    animate_datapack(dds6_h5parm,os.path.join(working_dir, 'tec_screen_plots'), num_processes=ncpu,
                     solset=deployment.screen_solset,
                     observable='tec', vmin=-60., vmax=60.,labels_in_radec=True,plot_crosses=False,phase_wrap=False,
                     overlay_solset='directionally_referenced')

    animate_datapack(dds6_h5parm, os.path.join(working_dir, 'amplitude_screen_plots'), num_processes=ncpu,
                     solset=deployment.screen_solset,
                     observable='amplitude', vmin=0.5, vmax=2., labels_in_radec=True, plot_crosses=False,
                     phase_wrap=False,
                     )

    # animate_datapack(merged_h5parm, os.path.join(working_dir, 'const_screen_plots'), num_processes=ncpu,
    #                  solset=deployment.screen_solset,
    #                  observable='const', vmin=-np.pi, vmax=np.pi, labels_in_radec=True, plot_crosses=False, phase_wrap=True)


def test_deployment():
    main(data_dir='/home/albert/store/lockman/test/root/L667218/subtract',
         working_dir='/home/albert/store/lockman/test/root/L667218/infer_screen',
         obs_num=667218,
         ref_dir=0,
         deployment_type='directional',
         block_size=10,
         ref_image_fits='/home/albert/store/lockman/lotss_archive_deep_image.app.restored.fits')


def add_args(parser):
    parser.add_argument('--obs_num', help='Obs number L*',
                        default=None, type=int, required=True)
    parser.add_argument('--data_dir', help='Where are the ms files are stored.',
                        default=None, type=str, required=True)
    parser.add_argument('--working_dir', help='Where to perform the imaging.',
                        default=None, type=str, required=True)
    parser.add_argument('--ref_dir', help='The index of reference dir.',
                        default=0, type=int, required=False)
    parser.add_argument('--ncpu', help='Number of CPUs.',
                        default=32, type=int, required=True)
    parser.add_argument('--deployment_type', help='The type of screen [directional, non_integral, tomographic].',
                        default='directional', type=str, required=False)
    parser.add_argument('--block_size', help='The number of time steps to process at once.',
                        default=10, type=int, required=False)
    parser.add_argument('--max_N', help='The maximum number of screen directions.',
                        default=250, type=int, required=False)
    parser.add_argument('--ref_image_fits', help='The Gaussian source list of the field used to choose locations of screen points.',
                        type=str, required=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Infers the value of DDTEC and a constant over a screen.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_args(parser)
    flags, unparsed = parser.parse_known_args()
    print("Running with:")
    for option, value in vars(flags).items():
        print("\t{} -> {}".format(option, value))
    main(**vars(flags))
