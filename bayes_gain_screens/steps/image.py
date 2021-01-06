import os
import glob
import sys
import argparse
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import subprocess
import logging

logger = logging.getLogger(__name__)


def cmd_call(cmd):
    logger.info("{}".format(cmd))
    exit_status = subprocess.call(cmd, shell=True)
    if exit_status:
        raise ValueError("Failed to  run: {}".format(cmd))


def flatten(f):
    """ Flatten a fits file so that it becomes a 2D image. Return new header and data """

    naxis = f[0].header['NAXIS']
    if naxis < 2:
        raise ValueError('Cannot make map from this')
    if naxis == 2:
        return fits.PrimaryHDU(header=f[0].header, data=f[0].data)

    w = WCS(f[0].header)
    wn = WCS(naxis=2)

    wn.wcs.crpix[0] = w.wcs.crpix[0]
    wn.wcs.crpix[1] = w.wcs.crpix[1]
    wn.wcs.cdelt = w.wcs.cdelt[0:2]
    wn.wcs.crval = w.wcs.crval[0:2]
    wn.wcs.ctype[0] = w.wcs.ctype[0]
    wn.wcs.ctype[1] = w.wcs.ctype[1]

    header = wn.to_header()
    header["NAXIS"] = 2
    copy = ('EQUINOX', 'EPOCH', 'BMAJ', 'BMIN', 'BPA', 'RESTFRQ', 'TELESCOP', 'OBSERVER')
    for k in copy:
        r = f[0].header.get(k)
        if r is not None:
            header[k] = r

    slice = []
    for i in range(naxis, 0, -1):
        if i <= 2:
            slice.append(np.s_[:], )
        else:
            slice.append(0)

    hdu = fits.PrimaryHDU(header=header, data=f[0].data[tuple(slice)])
    return hdu


def prepare_imaging(obs_num, data_dir, working_dir, mask, delete_ddfcache):
    data_dir = os.path.abspath(data_dir)
    ddfcache = glob.glob(os.path.join(working_dir, '*.ddfcache'))
    if len(ddfcache) > 0 and delete_ddfcache:
        logger.info("Deleting existing ddf cache")
        cleanup_working_dir(working_dir)
    logger.info("Preparing mslist")
    msfiles = glob.glob(os.path.join(data_dir, 'L{}*.ms'.format(obs_num)))
    if len(msfiles) == 0:
        raise IOError("No msfiles")
    mslist_file = os.path.join(working_dir, 'mslist.txt')
    with open(mslist_file, 'w') as f:
        for ms in msfiles:
            f.write("{}\n".format(ms))
    logger.info("Checking mask")
    if not os.path.isfile(mask):
        raise IOError("Couldn't find a mask matching {}".format(mask))
    return data_dir, working_dir, mslist_file, mask


def build_image_cmd(working_dir, template, **kwargs):
    with open(template, 'r') as f:
        cmd = f.read().format(**kwargs)
    logger.info(cmd)
    instruct = os.path.join(working_dir, 'instruct.sh')
    with open(instruct, 'w') as f:
        f.write(cmd)
    return cmd


def run_imaging(obs_num, data_dir, working_dir, script_dir, template_name, **kwargs):
    data_dir, working_dir, mslist_file, mask = prepare_imaging(obs_num=obs_num,
                                                               data_dir=data_dir,
                                                               working_dir=working_dir,
                                                               mask=kwargs['mask'],
                                                               delete_ddfcache=True)
    cmd = build_image_cmd(working_dir, os.path.join(script_dir, 'templates', template_name), **kwargs)
    cmd_call(cmd)


def cleanup_working_dir(working_dir):
    logger.info("Deleting cache since we're done.")
    for f in glob.glob(os.path.join(working_dir, "*.ddfcache")):
        cmd_call("rm -r {}".format(f))


def add_args(parser):
    parser.register("type", "bool", lambda v: v.lower() == "true")

    parser.add_argument('--image_type', help='Encoding of image type solset_alias:data_source. solset_alias can be: '
                                             'dirty, dds4, smoothed, smoothed_slow, screen, screen_slow.'
                                             'data source can be: subtracted, data, restricted.', required=True, type=str)
    parser.add_argument('--ncpu', help='Number of cpu to use, default=34', default=32, type=int)
    parser.add_argument('--obs_num', help='Obs number L*',
                        default=None, type=int, required=True)
    parser.add_argument('--data_dir', help='Where are the ms files are stored.',
                        default=None, type=str, required=True)
    parser.add_argument('--working_dir', help='Where to perform the imaging.',
                        default=None, type=str, required=True)
    parser.add_argument('--script_dir', help='Where the scripts are stored, "template" folder inside.',
                        default=None, type=str, required=True)
    parser.add_argument('--use_init_dico', help='Whether to initialise clean with dico.',
                        default=False, type="bool", required=False)
    parser.add_argument('--init_dico', help='Dico name (inside data dir) to initialise with.',
                        default=None, type=str, required=False)


def main(image_type, obs_num, data_dir, working_dir, script_dir, ncpu, use_init_dico, init_dico):
    logger.info("Changing to {}".format(working_dir))
    os.chdir(working_dir)
    kwargs = dict()

    kwargs['fluxthreshold'] = 70e-6
    kwargs['solsdir'] = os.path.join(data_dir, 'SOLSDIR')
    kwargs['peak_factor'] = 0.001
    kwargs['nfacets'] = 11
    kwargs['robust'] = -0.5
    kwargs['npix'] = 20000
    kwargs['ncpu'] = ncpu
    kwargs['mask'] = os.path.join(data_dir, 'image_full_ampphase_di_m.NS.mask01.fits')
    kwargs['output_name'] = "L{}_{}".format(obs_num, os.path.basename(working_dir))

    if init_dico is None:
        init_dico = os.path.join(data_dir, 'image_full_ampphase_di_m.NS.DicoModel')
    else:
        if not os.path.isfile(init_dico):
            raise ValueError("Supplied {} doesn't exist".format(init_dico))

    if os.path.isfile(init_dico) and use_init_dico:
        logger.info("Using {} to start clean.".format(init_dico))
        kwargs['init_dico'] = init_dico
        kwargs['major_iters'] = 1
    else:
        kwargs['major_iters'] = 5

    image_type = image_type.split(":")
    solset = image_type[0]
    data_source = image_type[1]
    if len(image_type) == 2:
        kwargs['weight_col'] = "IMAGING_WEIGHT"
    else:
        weight_col = image_type[2]
        if weight_col == 'imaging_weight':
            kwargs['weight_col'] = "IMAGING_WEIGHT"
        elif weight_col == 'outliers_flagged':
            kwargs['weight_col'] = "OUTLIERS_FLAGGED"
        else:
            raise ValueError("Invalid weight_col {}".format(weight_col))
    if solset == 'dirty':
        kwargs['major_iters'] = 0
        template_name = 'image_dirty_template'
    elif solset == 'dds4':
        kwargs['sols'] = 'DDS4_full'
        template_name = 'image_kms_sols_template'
    elif solset == 'smoothed':
        merged_h5parm = os.path.join(data_dir, 'L{}_{}_merged.h5'.format(obs_num, 'DDS5_full'))
        kwargs['sols'] = '{}:sol000/phase000+amplitude000'.format(merged_h5parm)
        template_name = 'image_h5parm_template'
    elif solset == 'screen':
        merged_h5parm = os.path.join(data_dir, 'L{}_{}_merged.h5'.format(obs_num, 'DDS6_full'))
        kwargs['sols'] = '{}:sol000/phase000+amplitude000'.format(merged_h5parm)
        template_name = 'image_h5parm_template'
    elif solset == 'smoothed_slow':
        merged_h5parm = os.path.join(data_dir, 'L{}_{}_merged.h5'.format(obs_num, 'DDS8_full'))
        kwargs['sols'] = '{}:smoothed_slow/phase000+amplitude000'.format(merged_h5parm)
        template_name = 'image_h5parm_template'
    elif solset == 'screen_slow':
        merged_h5parm = os.path.join(data_dir, 'L{}_{}_merged.h5'.format(obs_num, 'DDS8_full'))
        kwargs['sols'] = '{}:screen_slow/phase000+amplitude000'.format(merged_h5parm)
        template_name = 'image_h5parm_template'
    else:
        raise ValueError("Solset choice {} invalid".format(solset))

    if data_source == 'subtracted':
        kwargs['data_column'] = 'DATA_SUB'
    elif data_source == 'data':
        kwargs['data_column'] = 'DATA'
    elif data_source == 'restricted':
        kwargs['data_column'] = 'DATA_RESTRICTED'
        kwargs['npix'] = 10000
        kwargs['mask'] = os.path.join(data_dir, 'image_full_ampphase_di_m.NS.mask01.restricted.fits')
    elif data_source == 'restricted_full':
        kwargs['data_column'] = 'DATA_RESTRICTED'
    else:
        raise ValueError("Data source choice {} invalid".format(data_source))

    if kwargs.get('init_dico', False):
        template_name = template_name.replace("_template", "_restart_template")

    run_imaging(obs_num, data_dir, working_dir, script_dir=script_dir, template_name=template_name, **kwargs)
    cleanup_working_dir(working_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Image with kms sols or h5parm.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_args(parser)
    flags, unparsed = parser.parse_known_args()
    logger.info("Running with:")
    for option, value in vars(flags).items():
        logger.info("    {} -> {}".format(option, value))
    main(**vars(flags))
