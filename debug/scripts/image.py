import os
import sys
import glob
import argparse
import pylab as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import pyregion



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


def plot_image(filename, save_name=None, PPD=5, fig_size=20., background_mean=None, noise=None, radec_pix_center=None,
               radius_pix=None, ds9_regions=None):
    """
    Plot radio image.

    :param filename: str
        Fits file to plot
    :param save_name: str or list
        file name(s) with appendix
    :param PPD: float
        pix per dot (compress image be this much)
    :param fig_size: float
        fig size in inches
    :param background_mean: float
        value in muJy/beam
    :param noise: float
        value in muJy/beam
    :param radec_pix_center: tuple(float,float)
        Pixels of point to focus on (from ds9)
    :param radius_pix: float
        Pixels radius
    :param ds9_regions: str
        File with regions to plot
    """
    print("Plotting {}".format(filename))
    with fits.open(filename) as f:
        hdu = flatten(f)
        data = hdu.data
        npix = np.max(data.shape)

        dpi = (npix / PPD) / fig_size

        if background_mean is None or noise is None:
            ###
            # cheap noise floor
            data0 = np.copy(data)
            for _ in range(5):
                data0 = np.where(data - np.nanmean(data0) < 3. * np.nanstd(data0), data, np.nan)
            noise = np.nanstd(data0)
            background_mean = np.nanmean(data0)
            print("Found backgroud mean: {:.2f}muJy/beam, and background std: {:.2f}muJy/beam".format(
                background_mean * 1e6, noise * 1e6))
        else:
            background_mean = background_mean / 1e6
            noise = noise / 1e6
            print("Using backgroud mean: {:.2f}muJy/beam, and background std: {:.2f}muJy/beam".format(
                background_mean * 1e6, noise * 1e6))

        vmin = background_mean - 5. * noise  # np.percentile(data, 20.)
        vmax = background_mean + 50. * noise  # np.percentile(data, 80.)*10.

        wcs = WCS(hdu.header)
        fig = plt.figure(figsize=(fig_size, fig_size))
        ax = plt.subplot(projection=wcs)
        ax.imshow(np.sign(data) * np.sqrt(np.sign(data) * data),
                  vmin=np.sign(vmin) * np.sqrt(np.sign(vmin) * vmin),
                  vmax=np.sign(vmax) * np.sqrt(np.sign(vmax) * vmax),
                  origin='lower', cmap='bone_r')
        if radec_pix_center is not None:
            print("Centering on {} to radius {}".format(radec_pix_center, radius_pix))
            ax.set_xlim(radec_pix_center[0] - radius_pix, radec_pix_center[0] + radius_pix)
            ax.set_ylim(radec_pix_center[1] - radius_pix, radec_pix_center[1] + radius_pix)
        ax.coords.grid(True, color='black', ls='solid')
        ax.coords[0].set_axislabel('Right Ascension (J2000)')
        ax.coords[1].set_axislabel('Declination (J2000)')
        if ds9_regions is not None:
            print("Adding ds9 regions: {}".format(ds9_regions))
            r = pyregion.open(ds9_regions).as_imagecoord(hdu.header)
            patch_list, text_list = r.get_mpl_patches_texts()
            for p in patch_list:
                ax.add_patch(p)
            for t in text_list:
                ax.add_artist(t)
        if save_name is not None:
            if isinstance(save_name, (tuple, list)):
                for s in save_name:
                    plt.savefig(s, dpi=dpi, bbox_inches='tight', pad_inches=0)
                    print("Saved to {}".format(s))
            else:
                plt.savefig(save_name, dpi=dpi, bbox_inches='tight', pad_inches=0)
                print("Saved to {}".format(save_name))
            plt.close('all')


def image_dirty(obs_num, data_dir, working_dir, script_dir, **kwargs):
    data_dir, working_dir, mslist_file, mask = prepare_imaging(obs_num=obs_num,
                                                               data_dir=data_dir,
                                                               working_dir=working_dir,
                                                               mask='image_full_ampphase_di_m.NS.mask01.fits',
                                                               delete_ddfcache=True)
    kwargs['output_name'] = os.path.basename(working_dir)
    kwargs['mask'] = mask
    kwargs['fluxthreshold'] = 100e-6
    kwargs['major_iters'] = 0
    cmd = build_image_cmd(working_dir, os.path.join(script_dir, 'templates', 'image_dirty_template'), **kwargs)
    os.system(cmd)

    images = glob.glob(os.path.join(working_dir, "{}.app.restored.fits".format(kwargs['output_name'])))
    if len(images) == 0:
        raise ValueError("No image found to plot")
    plot_image(images[0], os.path.join(working_dir, "{}.app.restored.png".format(kwargs['output_name'])))


def image_DDS4(obs_num, data_dir, working_dir, script_dir, **kwargs):
    data_dir, working_dir, mslist_file, mask = prepare_imaging(obs_num=obs_num,
                                                               data_dir=data_dir,
                                                               working_dir=working_dir,
                                                               mask='image_full_ampphase_di_m.NS.mask01.fits',
                                                               delete_ddfcache=True)
    kwargs['output_name'] = os.path.basename(working_dir)
    kwargs['mask'] = mask
    kwargs['fluxthreshold'] = 100e-6
    kwargs['major_iters'] = 5
    kwargs['sols'] = 'DDS4_full'
    kwargs['solsdir'] = os.path.join(data_dir, 'SOLSDIR')

    if 'init_dico' in kwargs.keys():
        kwargs['major_iters'] = 1
        cmd = build_image_cmd(working_dir, os.path.join(script_dir, 'templates', 'image_kms_sols_restart_template'),
                              **kwargs)
    else:
        cmd = build_image_cmd(working_dir, os.path.join(script_dir, 'templates', 'image_kms_sols_template'), **kwargs)
    os.system(cmd)

    images = glob.glob(os.path.join(working_dir, "{}.app.restored.fits".format(kwargs['output_name'])))
    if len(images) == 0:
        raise ValueError("No image found to plot")
    plot_image(images[0], os.path.join(working_dir, "{}.app.restored.png".format(kwargs['output_name'])))


def image_smoothed(obs_num, data_dir, working_dir, script_dir, **kwargs):
    data_dir, working_dir, mslist_file, mask = prepare_imaging(obs_num=obs_num,
                                                               data_dir=data_dir,
                                                               working_dir=working_dir,
                                                               mask='image_full_ampphase_di_m.NS.mask01.fits',
                                                               delete_ddfcache=True)
    kwargs['output_name'] = os.path.basename(working_dir)
    kwargs['mask'] = mask
    kwargs['fluxthreshold'] = 100e-6
    kwargs['major_iters'] = 5
    merged_sol = os.path.join(data_dir, 'L{}_{}_merged.h5'.format(obs_num, 'DDS4_full'))
    kwargs['sols'] = '{}:smoothed000/phase000+amplitude000'.format(merged_sol)
    if 'init_dico' in kwargs.keys():
        kwargs['major_iters'] = 1
        cmd = build_image_cmd(working_dir, os.path.join(script_dir, 'templates', 'image_h5parm_restart_template'),
                              **kwargs)
    else:
        cmd = build_image_cmd(working_dir, os.path.join(script_dir, 'templates', 'image_h5parm_template'), **kwargs)
    os.system(cmd)

    images = glob.glob(os.path.join(working_dir, "{}.app.restored.fits".format(kwargs['output_name'])))
    if len(images) == 0:
        raise ValueError("No image found to plot")
    plot_image(images[0], os.path.join(working_dir, "{}.app.restored.png".format(kwargs['output_name'])))


def image_smoothed_slow(obs_num, data_dir, working_dir, script_dir, **kwargs):
    data_dir, working_dir, mslist_file, mask = prepare_imaging(obs_num=obs_num,
                                                               data_dir=data_dir,
                                                               working_dir=working_dir,
                                                               mask='image_full_ampphase_di_m.NS.mask01.fits',
                                                               delete_ddfcache=True)
    kwargs['output_name'] = os.path.basename(working_dir)
    kwargs['mask'] = mask
    kwargs['fluxthreshold'] = 0.
    kwargs['major_iters'] = 5
    merged_sol = os.path.join(data_dir, 'L{}_{}_merged.h5'.format(obs_num, 'DDS4_full'))
    kwargs['sols'] = '{}:smoothed_slow000/phase000+amplitude000'.format(merged_sol)
    if 'init_dico' in kwargs.keys():
        kwargs['major_iters'] = 1
        cmd = build_image_cmd(working_dir, os.path.join(script_dir, 'templates', 'image_h5parm_restart_template'),
                              **kwargs)
    else:
        cmd = build_image_cmd(working_dir, os.path.join(script_dir, 'templates', 'image_h5parm_template'), **kwargs)
    os.system(cmd)

    images = glob.glob(os.path.join(working_dir, "{}.app.restored.fits".format(kwargs['output_name'])))
    if len(images) == 0:
        raise ValueError("No image found to plot")
    plot_image(images[0], os.path.join(working_dir, "{}.app.restored.png".format(kwargs['output_name'])))


def image_screen(obs_num, data_dir, working_dir, script_dir, **kwargs):
    data_dir, working_dir, mslist_file, mask = prepare_imaging(obs_num=obs_num,
                                                               data_dir=data_dir,
                                                               working_dir=working_dir,
                                                               mask='image_full_ampphase_di_m.NS.mask01.fits',
                                                               delete_ddfcache=True)

    kwargs['output_name'] = os.path.basename(working_dir)
    kwargs['mask'] = mask
    kwargs['fluxthreshold'] = 0.
    kwargs['major_iters'] = 5
    merged_h5parm = os.path.join(data_dir, 'L{}_{}_merged.h5'.format(obs_num, 'DDS4_full'))
    kwargs['sols'] = '{}:screen_posterior/phase000+amplitude000'.format(merged_h5parm)
    if 'init_dico' in kwargs.keys():
        kwargs['major_iters'] = 1
        cmd = build_image_cmd(working_dir, os.path.join(script_dir, 'templates', 'image_h5parm_restart_template'),
                              **kwargs)
    else:
        cmd = build_image_cmd(working_dir, os.path.join(script_dir, 'templates', 'image_h5parm_template'), **kwargs)
    os.system(cmd)

    images = glob.glob(os.path.join(working_dir, "{}.app.restored.fits".format(kwargs['output_name'])))
    if len(images) == 0:
        raise ValueError("No image found to plot")
    plot_image(images[0], os.path.join(working_dir, "{}.app.restored.png".format(kwargs['output_name'])))


def image_screen_slow(obs_num, data_dir, working_dir, script_dir, **kwargs):
    data_dir, working_dir, mslist_file, mask = prepare_imaging(obs_num=obs_num,
                                                               data_dir=data_dir,
                                                               working_dir=working_dir,
                                                               mask='image_full_ampphase_di_m.NS.mask01.fits',
                                                               delete_ddfcache=True)

    kwargs['output_name'] = os.path.basename(working_dir)
    kwargs['mask'] = mask
    kwargs['fluxthreshold'] = 0.
    kwargs['major_iters'] = 5
    merged_sol = os.path.join(data_dir, 'L{}_{}_merged.h5'.format(obs_num, 'DDS4_full'))
    kwargs['sols'] = '{}:screen_slow000/phase000+amplitude000'.format(merged_sol)

    if 'init_dico' in kwargs.keys():
        kwargs['major_iters'] = 1
        cmd = build_image_cmd(working_dir, os.path.join(script_dir, 'templates', 'image_h5parm_restart_template'), **kwargs)
    else:
        cmd = build_image_cmd(working_dir, os.path.join(script_dir, 'templates', 'image_h5parm_template'), **kwargs)
    os.system(cmd)

    images = glob.glob(os.path.join(working_dir, "{}.app.restored.fits".format(kwargs['output_name'])))
    if len(images) == 0:
        raise ValueError("No image found to plot")
    plot_image(images[0], os.path.join(working_dir, "{}.app.restored.png".format(kwargs['output_name'])))


def build_image_cmd(working_dir, template, **kwargs):
    with open(template, 'r') as f:
        cmd = f.read().format(**kwargs)
    print(cmd)
    instruct = os.path.join(working_dir, 'instruct.sh')
    with open(instruct, 'w') as f:
        f.write(cmd)
    return cmd


def prepare_imaging(obs_num, data_dir, working_dir, mask, delete_ddfcache):
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
    return data_dir, working_dir, mslist_file, mask


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
    parser.add_argument('--script_dir', help='Where the scripts are stored.',
                        default=None, type=str, required=True)
    parser.add_argument('--use_init_dico', help='Whether to initialise clean with dico.',
                        default=False, type="bool", required=False)


def main(image_type, obs_num, data_dir, working_dir, script_dir, ncpu, use_init_dico):
    kwargs = {}
    kwargs['peak_factor'] = 0.001
    kwargs['nfacets'] = 11
    kwargs['robust'] = -0.5
    init_dico = os.path.join(data_dir, 'image_full_ampphase_di_m.NS.DicoModel')
    if os.path.isfile(init_dico) and use_init_dico:
        kwargs['init_dico'] = init_dico
    if image_type == 'image_subtract_dirty':
        image_dirty(obs_num=obs_num,
                    data_dir=data_dir, working_dir=working_dir, ncpu=ncpu, script_dir=script_dir,
                    data_column='DATA_SUB')
    if image_type == 'image_smoothed':
        image_smoothed(obs_num, data_dir, working_dir, ncpu=ncpu, script_dir=script_dir, data_column='DATA', **kwargs)
    if image_type == 'image_smoothed_slow':
        image_smoothed_slow(obs_num, data_dir, working_dir, ncpu=ncpu, script_dir=script_dir, data_column='DATA', **kwargs)
    if image_type == 'image_screen':
        image_screen(obs_num, data_dir, working_dir, ncpu=ncpu, script_dir=script_dir, data_column='DATA', **kwargs)
    if image_type == 'image_screen_slow':
        image_screen_slow(obs_num, data_dir, working_dir, ncpu=ncpu, script_dir=script_dir, data_column='DATA', **kwargs)
    if image_type == 'image_dds4':
        image_DDS4(obs_num, data_dir, working_dir, ncpu=ncpu, script_dir=script_dir, data_column='DATA', **kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Image with kms sols or h5parm.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_args(parser)
    flags, unparsed = parser.parse_known_args()
    print("Running with:")
    for option, value in vars(flags).items():
        print("    {} -> {}".format(option, value))
    try:
        main(**vars(flags))
        exit(0)
    except:
        exit(1)
