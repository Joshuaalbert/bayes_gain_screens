from astropy.io import fits
import astropy.coordinates as ac
import astropy.units as au
from astropy import wcs
import matplotlib as mpl
mpl.use('Agg')
import pylab as plt
import numpy as np
from matplotlib.patches import Circle
import argparse
import os

def great_circle_sep(ra1, dec1, ra2, dec2):
    dra = np.abs(ra1 - ra2)
    # ddec = np.abs(dec1-dec2)
    num2 = (np.cos(dec2) * np.sin(dra)) ** 2 + (
                np.cos(dec1) * np.sin(dec2) - np.sin(dec1) * np.cos(dec2) * np.cos(dra)) ** 2
    den = np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(dra)
    return np.arctan2(np.sqrt(num2), den)


def get_screen_directions_from_image(image_fits, flux_limit=0.1, max_N=None, min_spacing_arcmin=1., plot=False,
                                     seed_directions=None, fill_in_distance=None,
                                     fill_in_flux_limit=0.):
    """Given a srl file containing the sources extracted from the apparent flux image of the field,
    decide the screen directions

    :param srl_fits: str
        The path to the srl file, typically created by pybdsf
    :return: float, array [N, 2]
        The `N` sources' coordinates as an ``astropy.coordinates.ICRS`` object
    """



def get_screen_directions(ref_image_fits, flux_limit=0.1, max_N=None, min_spacing_arcmin=1.,
                                     seed_directions=None, fill_in_distance=None,
                                     fill_in_flux_limit=0., working_dir=None):
    """Given a srl file containing the sources extracted from the apparent flux image of the field,
    decide the screen directions

    :param srl_fits: str
        The path to the srl file, typically created by pybdsf
    :return: float, array [N, 2]
        The `N` sources' coordinates as an ``astropy.coordinates.ICRS`` object
    """
    print("Getting screen directions from image.")

    with fits.open(ref_image_fits) as hdul:
        # ra,dec, _, freq
        data = hdul[0].data
        w = wcs.WCS(hdul[0].header)
        #         Nra, Ndec,_,_ = data.shape
        where_limit = np.where(data >= flux_limit)
        arg_sort = np.argsort(data[where_limit])[::-1]

        ra = []
        dec = []
        f = []
        sizes = []
        if seed_directions is not None:
            print("Using seed directions.")
            ra = list(seed_directions[:, 0])
            dec = list(seed_directions[:, 1])
            f = list(flux_limit * np.ones(len(ra)))
            sizes = list(120*np.ones(len(ra)))
        idx = []
        for i in arg_sort:
            if max_N is not None:
                if len(ra) >= max_N:
                    break
            pix = [where_limit[3][i], where_limit[2][i], where_limit[1][i], where_limit[0][i]]
            #             logging.info("{} -> {}".format(i, pix))
            #             pix = np.reshape(np.array(np.unravel_index(i, [Nra, Ndec, 1, 1])), (1, 4))
            coords = w.wcs_pix2world([pix], 1)  # degrees
            ra_ = coords[0, 0] * np.pi / 180.
            dec_ = coords[0, 1] * np.pi / 180.

            if len(ra) == 0:
                ra.append(ra_)
                dec.append(dec_)
                f.append(data[pix[3], pix[2], pix[1], pix[0]])
                print(
                    "Auto-append first: Found {} at {} {}".format(f[-1], ra[-1] * 180. / np.pi, dec[-1] * 180. / np.pi))

                idx.append(i)
                sizes.append(120.)
                continue
            dist = great_circle_sep(np.array(ra), np.array(dec), ra_, dec_) * 180. / np.pi
            if np.all(dist > min_spacing_arcmin / 60.):
                ra.append(ra_)
                dec.append(dec_)
                f.append(data[pix[3], pix[2], pix[1], pix[0]])
                sizes.append(120.)
                print("Found {} at {} {}".format(f[-1], ra[-1] * 180. / np.pi, dec[-1] * 180. / np.pi))
                idx.append(i)

        first_found = len(idx)

        if fill_in_distance is not None:
            where_limit = np.where(np.logical_and(data < np.min(f), data >= fill_in_flux_limit))
            arg_sort = np.argsort(data[where_limit])[::-1]
            # use remaining brightest sources to get fillers
            for i in arg_sort:
                if max_N is not None:
                    if len(ra) >= max_N:
                        break
                pix = [where_limit[3][i], where_limit[2][i], where_limit[1][i], where_limit[0][i]]
                #                 logging.info("{} -> {}".format(i, pix))
                coords = w.wcs_pix2world([pix], 1)  # degrees

                ra_ = coords[0, 0] * np.pi / 180.
                dec_ = coords[0, 1] * np.pi / 180.

                dist = great_circle_sep(np.array(ra), np.array(dec), ra_, dec_) * 180. / np.pi
                if np.all(dist > fill_in_distance / 60.):
                    ra.append(ra_)
                    dec.append(dec_)
                    f.append(data[pix[3], pix[2], pix[1], pix[0]])
                    sizes.append(0.5*np.men(dist)*3600.)
                    print(
                        "Found filler {} at {} {}".format(f[-1], ra[-1] * 180. / np.pi, dec[-1] * 180. / np.pi))
                    idx.append(i)

        if max_N is not None:
            f = np.array(f)[:max_N]
            ra = np.array(ra)[:max_N]
            dec = np.array(dec)[:max_N]
            sizes = list(np.array(sizes)[:max_N])
    f = np.array(f)
    ra = np.array(ra)
    dec = np.array(dec)
    sizes = list(sizes)
    # plotting
    plt.scatter(ra, dec, c=np.linspace(0., 1., len(ra)), cmap='jet', s=np.sqrt(10000. * f), alpha=1.)
    target = Circle((np.mean(ra)*180/np.pi, np.mean(dec)*180/np.pi), radius=3.56 / 2., fc=None, alpha=0.2)
    ax = plt.gca()
    ax.add_patch(target)
    target = Circle((np.mean(ra)*180/np.pi, np.mean(dec)*180/np.pi), radius=4.75 / 2., fc=None, alpha=0.2)
    ax = plt.gca()
    ax.add_patch(target)
    plt.title("Brightest {} sources".format(len(f)))
    plt.xlabel('ra (deg)')
    plt.xlabel('dec (deg)')
    plt.savefig(os.path.join(working_dir, 'calibrators.png'))
    plt.close('all')
    print("Found {} sources.".format(len(ra)))
    if seed_directions is not None:
        ra = list(seed_directions[:, 0]) + list(ra)
        dec = list(seed_directions[:, 1]) + list(dec)
    return ac.ICRS(ra=ra * au.rad, dec=dec * au.rad), sizes


def write_reg_file(filename, radius_arcsec, directions, color='green'):
    if not isinstance(radius_arcsec, (list, tuple)):
        radius_arcsec = [radius_arcsec] * len(directions)
    with open(filename, 'w') as f:
        f.write('# Region file format: DS9 version 4.1\n')
        f.write(
            'global color={color} dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n'.format(
                color=color))
        f.write('fk5\n')
        for r, d in zip(radius_arcsec, directions):
            f.write('circle({},{},{}")\n'.format(
                d.ra.to_string(unit=au.hour, sep=(":", ":"), alwayssign=False, precision=3),
                d.dec.to_string(unit=au.deg, sep=(":", ":"), alwayssign=True, precision=2), r))
        print("Wrote calibrators to file: {}".format(filename))


def add_args(parser):
    parser.register("type", "bool", lambda v: v.lower() == "true")

    parser.add_argument('--working_dir', default='./', help='Where to store things like output', required=False,
                        type=str)
    parser.add_argument('--region_file', help='boxfile, required argument', required=True, type=str)
    parser.add_argument('--ref_image_fits',
                        help='image of field.',
                        type=str, required=True)
    parser.add_argument('--flux_limit', help='Peak flux cut off for source selection.',
                        default=0.15, type=float)
    parser.add_argument('--max_N', help='Max num of sources',
                        default=None, type=int, required=False)

    parser.add_argument('--min_spacing_arcmin', help='Min distance in arcmin of sources.',
                        default=10., type=float, required=False)
    parser.add_argument('--fill_in_distance',
                        help='If not None then uses fainter sources to fill in some large areas further than fill_in_distance from nearest selected source in arcmin.',
                        default=None, type=float, required=False)
    parser.add_argument('--fill_in_flux_limit',
                        help='If fill_in_distance is not None then this is the secondary peak flux cutoff for fill in sources.',
                        default=None, type=float, required=False)


def main(working_dir, region_file, ref_image_fits,
         flux_limit, max_N, min_spacing_arcmin, fill_in_distance, fill_in_flux_limit):
    directions, sizes = get_screen_directions(ref_image_fits=ref_image_fits,
                                              flux_limit=flux_limit, max_N=max_N, min_spacing_arcmin=min_spacing_arcmin,
                                              fill_in_distance=fill_in_distance, fill_in_flux_limit=fill_in_flux_limit,
                                              working_dir=working_dir)
    write_reg_file(region_file, sizes, directions, 'red')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Select calibrator sources.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_args(parser)
    flags, unparsed = parser.parse_known_args()
    print("Running with:")
    for option, value in vars(flags).items():
        print("    {} -> {}".format(option, value))

    main(**vars(flags))
