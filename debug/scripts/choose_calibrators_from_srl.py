from astropy.io import fits
import astropy.coordinates as ac
import astropy.units as au
import pylab as plt
import numpy as np
from matplotlib.patches import Circle
from scipy.spatial import cKDTree
import argparse
import os

def great_circle_sep(ra1, dec1, ra2, dec2):
    dra = np.abs(ra1 - ra2)
    # ddec = np.abs(dec1-dec2)
    num2 = (np.cos(dec2) * np.sin(dra)) ** 2 + (
                np.cos(dec1) * np.sin(dec2) - np.sin(dec1) * np.cos(dec2) * np.cos(dra)) ** 2
    den = np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(dra)
    return np.arctan2(np.sqrt(num2), den)


def get_screen_directions(srl_fits='/home/albert/lofar1_1/imaging/lockman_deep_archive.pybdsm.srl.fits', flux_limit=0.1,
                          max_N=None, min_spacing_arcmin=1., flux_type='Peak_flux', plot=False, fill_in_distance=None,
                          fill_in_flux_limit=0.):
    """Given a srl file containing the sources extracted from the apparent flux image of the field,
    decide the screen directions
    :param srl_fits: str
        The path to the srl file, typically created by pybdsf
    :return: float, array [N, 2]
        The `N` sources' coordinates as an ``astropy.coordinates.ICRS`` object
    """
    hdu = fits.open(srl_fits)
    data = hdu[1].data

    arg = np.argsort(data[flux_type])[::-1]

    ra = []
    dec = []
    idx = []
    for i in arg:
        if data[flux_type][i] < flux_limit:
            continue
        ra_ = data['RA'][i] * np.pi / 180.
        dec_ = data['DEC'][i] * np.pi / 180.

        if len(ra) == 0:
            ra.append(ra_)
            dec.append(dec_)
            idx.append(i)
            continue
        dist = great_circle_sep(ra_, dec_, np.array(ra), np.array(dec)) * 180. / np.pi
        if np.all(dist > min_spacing_arcmin / 60.):
            ra.append(ra_)
            dec.append(dec_)
            idx.append(i)
            continue

    first_found = len(idx)

    if fill_in_distance is not None:
        # use remaining brightest sources to get fillers
        for i in arg:
            if data[flux_type][i] < fill_in_flux_limit:
                continue
            ra_ = data['RA'][i] * np.pi / 180.
            dec_ = data['DEC'][i] * np.pi / 180.
            dist = great_circle_sep(ra_, dec_, np.array(ra), np.array(dec)) * 180. / np.pi
            if np.all(dist > fill_in_distance / 60.):
                print("Filling in with source of peak flux {}".format(data[flux_type][i]))
                ra.append(ra_)
                dec.append(dec_)
                idx.append(i)
                continue

    f = data[flux_type][idx]
    ra = data['RA'][idx]
    dec = data['DEC'][idx]
    c = data['S_code'][idx]

    sizes = np.ones(len(idx))
    sizes[:first_found] = 120.
    sizes[first_found:] = 240.

    if max_N is not None:
        arg = np.argsort(f)[::-1][:max_N]
        f = f[arg]
        ra = ra[arg]
        dec = dec[arg]
        c = c[arg]
    print('Found {} sources'.format(len(f)))
    if plot:
        plt.scatter(ra, dec, c=np.linspace(0., 1., len(ra)), cmap='jet', s=np.sqrt(10000. * f), alpha=1.)

        target = Circle((np.mean(ra), np.mean(dec)), radius=3.56 / 2., fc=None, alpha=0.2)
        ax = plt.gca()
        ax.add_patch(target)
        target = Circle((np.mean(ra), np.mean(dec)), radius=4.75 / 2., fc=None, alpha=0.2)
        ax = plt.gca()
        ax.add_patch(target)
        plt.title("Brightest {} sources".format(len(f)))
        plt.xlabel('ra (deg)')
        plt.xlabel('dec (deg)')
        #         plt.savefig("scren_directions.png")
        plt.show()
        interdist, _ = cKDTree(np.stack([ra, dec], axis=1) * 60.).query(np.stack([ra, dec], axis=1) * 60., k=2)
        interdist = interdist[:, 1]
        #         interdist = pdist(np.stack([ra,dec],axis=1)*60.)
        plt.hist(interdist, bins=len(f))
        plt.title("inter-facet distance distribution")
        plt.xlabel('inter-facet distance [arcmin]')
        #         plt.savefig("interfacet_distance_dist.png")
        plt.show()
    return ac.ICRS(ra=ra * au.deg, dec=dec * au.deg), list(sizes)


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


def add_args(parser):
    parser.register("type", "bool", lambda v: v.lower() == "true")

    parser.add_argument('--working_dir', default='./', help='Where to store things like output', required=False,
                        type=str)
    parser.add_argument('--region_file', help='boxfile, required argument', required=True, type=str)
    parser.add_argument('--srl_fits',
                        help='srl of field stored as srl fits',
                        type=str, required=True)
    parser.add_argument('--flux_limit', help='Peak flux cut off for source selection.',
                        default=0.15, type=float)
    parser.add_argument('--max_N', help='Max num of sources',
                        default=None, type=int, required=False)

    parser.add_argument('--min_spacing_arcmin', help='Min distance in arcmin of sources.',
                        default=10., type=float, required=False)
    parser.add_argument('--plot', help='Whether to plot.',
                        default=False, type="bool", required=False)
    parser.add_argument('--fill_in_distance',
                        help='If not None then uses fainter sources to fill in some areas further than fill_in_distance from nearest selected source in arcmin.',
                        default=None, type=float, required=False)
    parser.add_argument('--min_spacing_arcmin',
                        help='If fill_in_distance is not None then this is the secondary peak flux cutoff for fill in sources.',
                        default=0.05, type=float, required=False)


def main(working_dir, region_file, srl_fits,
         flux_limit, max_N, min_spacing_arcmin, plot, fill_in_distance, fill_in_flux_limit):
    region_file = os.path.join(os.path.abspath(working_dir), os.path.basename(region_file))
    directions, sizes = get_screen_directions(srl_fits=srl_fits,
                                              flux_limit=flux_limit, max_N=max_N, min_spacing_arcmin=min_spacing_arcmin,
                                              plot=plot,
                                              fill_in_distance=fill_in_distance, fill_in_flux_limit=fill_in_flux_limit)
    write_reg_file(region_file, sizes, directions, 'red')


def lockman_run():
    main(working_dir='./', region_file='LH_auto_select.reg', srl_fits='lockman_deep_archive.pybdsm.srl.fits',
         flux_limit=0.15, max_N=None, min_spacing_arcmin=10., plot=False, fill_in_distance=60.,
         fill_in_flux_limit=0.05)


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
