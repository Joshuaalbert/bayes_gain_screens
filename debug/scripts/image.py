import os
import glob
import argparse

TEMPLATE_FOLDER = '/home/albert/store/lockman/scripts/templates'


def image_dirty(obs_num, data_dir, working_dir, **kwargs):
    data_dir, working_dir, mslist_file, mask, clustercat = prepare_imaging(obs_num=obs_num,
                                                                           data_dir=data_dir,
                                                                           working_dir=working_dir,
                                                                           mask='image_full_ampphase_di_m.NS.mask01.fits',
                                                                           clustercat='subtract.ClusterCat.npy',
                                                                           delete_ddfcache=True)
    kwargs['output_name'] = os.path.basename(working_dir)
    kwargs['mask'] = mask
    kwargs['fluxthreshold'] = 100e-6
    kwargs['nfacets'] = 11
    kwargs['robust'] = -0.5
    kwargs['major_iters'] = 0
    cmd = build_image_cmd(working_dir, os.path.join(TEMPLATE_FOLDER, 'image_dirty_template'))
    os.system(cmd)


def image_DDS4(obs_num, data_dir, working_dir, **kwargs):
    data_dir, working_dir, mslist_file, mask, clustercat = prepare_imaging(obs_num=obs_num,
                                                                           data_dir=data_dir,
                                                                           working_dir=working_dir,
                                                                           mask='image_full_ampphase_di_m.NS.mask01.fits',
                                                                           clustercat='subtract.ClusterCat.npy',
                                                                           delete_ddfcache=True)
    kwargs['output_name'] = os.path.basename(working_dir)
    kwargs['mask'] = mask
    kwargs['fluxthreshold'] = 100e-6
    kwargs['nfacets'] = 11
    kwargs['robust'] = -0.5
    kwargs['major_iters'] = 5
    kwargs['sols'] = 'DDS4_full'
    kwargs['solsdir'] = os.path.join(data_dir, 'SOLSDIR')
    cmd = build_image_cmd(working_dir, os.path.join(TEMPLATE_FOLDER, 'image_kms_sols_template'))
    os.system(cmd)


def image_smoothed(obs_num, data_dir, working_dir, **kwargs):
    data_dir, working_dir, mslist_file, mask, clustercat = prepare_imaging(obs_num=obs_num,
                                                                           data_dir=data_dir,
                                                                           working_dir=working_dir,
                                                                           mask='image_full_ampphase_di_m.NS.mask01.fits',
                                                                           clustercat='subtract.ClusterCat.npy',
                                                                           delete_ddfcache=True)
    kwargs['output_name'] = os.path.basename(working_dir)
    kwargs['mask'] = mask
    kwargs['fluxthreshold'] = 100e-6
    kwargs['nfacets'] = 11
    kwargs['robust'] = -0.5
    kwargs['major_iters'] = 5
    merged_sol = os.path.join(data_dir, 'L{}_{}_merged.sols.npz'.format(obs_num, 'DDS4_full'))
    kwargs['sols'] = '{}:smoothed000/phase000+amplitude000'.format(merged_sol)
    cmd = build_image_cmd(working_dir, os.path.join(TEMPLATE_FOLDER, 'image_h5parm_template'))
    os.system(cmd)


def image_smoothed_slow(obs_num, data_dir, working_dir, **kwargs):
    data_dir, working_dir, mslist_file, mask, clustercat = prepare_imaging(obs_num=obs_num,
                                                                           data_dir=data_dir,
                                                                           working_dir=working_dir,
                                                                           mask='image_full_ampphase_di_m.NS.mask01.fits',
                                                                           clustercat='subtract.ClusterCat.npy',
                                                                           delete_ddfcache=True)
    kwargs['output_name'] = os.path.basename(working_dir)
    kwargs['mask'] = mask
    kwargs['fluxthreshold'] = 0.
    kwargs['nfacets'] = 11
    kwargs['robust'] = -0.5
    kwargs['major_iters'] = 7
    merged_sol = os.path.join(data_dir, 'L{}_{}_merged.sols.npz'.format(obs_num, 'DDS4_full'))
    kwargs['sols'] = '{}:smoothed_slow000/phase000+amplitude000'.format(merged_sol)
    cmd = build_image_cmd(working_dir, os.path.join(TEMPLATE_FOLDER, 'image_h5parm_template'))
    os.system(cmd)


def image_screen(obs_num, data_dir, working_dir, **kwargs):
    data_dir, working_dir, mslist_file, mask, clustercat = prepare_imaging(obs_num=obs_num,
                                                                           data_dir=data_dir,
                                                                           working_dir=working_dir,
                                                                           mask='image_full_ampphase_di_m.NS.mask01.fits',
                                                                           clustercat='subtract.ClusterCat.npy',
                                                                           delete_ddfcache=True)

    kwargs['output_name'] = os.path.basename(working_dir)
    kwargs['mask'] = mask
    kwargs['fluxthreshold'] = 0.
    kwargs['nfacets'] = 11
    kwargs['robust'] = -0.5
    kwargs['major_iters'] = 7
    merged_sol = os.path.join(data_dir, 'L{}_{}_merged.sols.npz'.format(obs_num, 'DDS4_full'))
    kwargs['sols'] = '{}:screen_posterior000/phase000+amplitude000'.format(merged_sol)
    cmd = build_image_cmd(working_dir, os.path.join(TEMPLATE_FOLDER, 'image_h5parm_template'))
    os.system(cmd)


def image_screen_slow(obs_num, data_dir, working_dir, **kwargs):
    data_dir, working_dir, mslist_file, mask, clustercat = prepare_imaging(obs_num=obs_num,
                                                                           data_dir=data_dir,
                                                                           working_dir=working_dir,
                                                                           mask='image_full_ampphase_di_m.NS.mask01.fits',
                                                                           clustercat='subtract.ClusterCat.npy',
                                                                           delete_ddfcache=True)

    kwargs['output_name'] = os.path.basename(working_dir)
    kwargs['mask'] = mask
    kwargs['fluxthreshold'] = 0.
    kwargs['nfacets'] = 11
    kwargs['robust'] = -0.5
    kwargs['major_iters'] = 7
    merged_sol = os.path.join(data_dir, 'L{}_{}_merged.sols.npz'.format(obs_num, 'DDS4_full'))
    kwargs['sols'] = '{}:screen_slow000/phase000+amplitude000'.format(merged_sol)
    cmd = build_image_cmd(working_dir, os.path.join(TEMPLATE_FOLDER, 'image_h5parm_template'))
    os.system(cmd)


def build_image_cmd(working_dir, template, **kwargs):
    with open(template, 'r') as f:
        cmd = f.read().format(**kwargs)
    print(cmd)
    instruct = os.path.join(working_dir, 'instruct.sh')
    with open(instruct, 'w') as f:
        f.write(cmd)
    return cmd


def prepare_imaging(obs_num, data_dir, working_dir, mask, clustercat, delete_ddfcache):
    data_dir = os.path.abspath(data_dir)
    print("Changing to {}".format(working_dir))
    os.chdir(working_dir)
    ddfcache = glob.glob(os.path.join(working_dir, '*.ddfcache'))
    if len(ddfcache) > 0 and delete_ddfcache:
        print("Deleting existing ddf cache")
        for d in ddfcache:
            os.unlink(d)
    print("Preparing mslist")
    msfiles = glob.glob(os.path.join(data_dir, 'L{}*.ms'.format(obs_num)))
    if len(msfiles) == 0:
        raise IOError("No msfiles")
    mslist_file = os.path.join(working_dir, 'mslist.txt')
    with open(mslist_file, 'w') as f:
        for ms in msfiles:
            f.write("{}\n".format(ms))
    print("Getting mask")
    mask = os.path.abspath(mask)
    if not os.path.isfile(mask):
        mask = os.path.join(data_dir, os.path.basename(mask))
        if not os.path.isfile(mask):
            raise IOError("Couldn't find a mask matching {}".format(mask))
    print("Getting clustercat")
    clustercat = os.path.abspath(clustercat)
    if not os.path.isfile(clustercat):
        clustercat = os.path.join(data_dir, os.path.basename(clustercat))
        if not os.path.isfile(clustercat):
            raise IOError("Couldn't find a clustercat matching {}".format(clustercat))
    return data_dir, working_dir, mslist_file, mask, clustercat


def add_args(parser):
    parser.register("type", "bool", lambda v: v.lower() == "true")

    parser.add_argument('--image_type', help='Type of image command', required=True, type=str)
    parser.add_argument('--ncpu', help='Number of cpu to use, default=34', default=32, type=int)
    parser.add_argument('--obs_num', help='Obs number L*',
                        default=None, type=int, required=True)
    parser.add_argument('--data_dir', help='Where are the ms files are stored.',
                        default=None, type=str, required=True)
    parser.add_argument('--working_dir', help='Where to perform the imaging.',
                        default=None, type=str, required=True)


def main(image_type, obs_num, data_dir, working_dir):
    if image_type == 'image_subtract_dirty':
        image_dirty(obs_num=obs_num,
                    data_dir=data_dir, working_dir=working_dir, data_column='DATA_SUB')
    if image_type == 'image_smoothed':
        image_smoothed(obs_num, data_dir, working_dir, data_column='DATA')
    if image_type == 'image_smoothed_slow':
        image_smoothed_slow(obs_num, data_dir, working_dir, data_column='DATA')
    if image_type == 'image_screen':
        image_screen(obs_num, data_dir, working_dir, data_column='DATA')
    if image_type == 'image_screen_slow':
        image_screen_slow(obs_num, data_dir, working_dir, data_column='DATA')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Keep soures inside box region, subtract everything else and create new ms',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_args(parser)
    flags, unparsed = parser.parse_known_args()
    print("Running with:")
    for option, value in vars(flags).items():
        print("    {} -> {}".format(option, value))
    main(**vars(flags))
