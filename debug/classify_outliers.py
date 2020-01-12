import numpy as np
from bayes_gain_screens.outlier_detection import filter_tec_dir
import glob, os
from bayes_gain_screens.datapack import DataPack
from bayes_gain_screens.misc import voronoi_finite_polygons_2d
import matplotlib
matplotlib.use('tkagg')
import pylab as plt
from scipy.spatial import Voronoi
from scipy.optimize import linprog
from astropy.io import fits
from astropy.wcs import WCS

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


def in_hull(points, x):
    n_points = len(points)
    n_dim = len(x)
    c = np.zeros(n_points)
    A = np.r_[points.T,np.ones((1,n_points))]
    b = np.r_[x, np.ones(1)]
    lp = linprog(c, A_eq=A, b_eq=b)
    return lp.success

def click_through(datapack, ref_image):
    with fits.open(ref_image) as f:
        hdu = flatten(f)
        data = hdu.data
        npix = np.max(data.shape)
        wcs = WCS(hdu.header)

    save_file = os.path.join('./outliers_{:03d}.npy'.format(len(glob.glob('./outliers_*.npy'))))

    dp = DataPack(datapack, readonly=True)

    dp.current_solset = 'directionally_referenced'
    dp.select(pol=slice(0, 1, 1))
    tec, axes = dp.tec
    _, Nd, Na, Nt = tec.shape
    tec_uncert, _ = dp.weights_tec
    _, directions = dp.get_directions(axes['dir'])
    directions = np.stack([directions.ra.deg, directions.dec.deg], axis=1)
    directions = wcs.wcs_world2pix(directions, 0)
    _, times = dp.get_times(axes['time'])
    times = times.mjd*86400.
    times -= times[0]
    times /= 3600.

    xmin = directions[:, 0].min()
    xmax = directions[:, 0].max()
    ymin = directions[:, 1].min()
    ymax = directions[:, 1].max()

    radius = xmax - xmin

    ref_dir = directions[0:1, :]

    _, guess_flags = filter_tec_dir(tec[0,...], directions, init_y_uncert=None, min_res=8.)
    # guess_flags = np.ones([Nd, Na, Nt], np.bool)
    human_flags = -1*np.ones([Nd, Na, Nt], np.int32)

    # compute Voronoi tesselation
    vor = Voronoi(directions)
    regions, vertices = voronoi_finite_polygons_2d(vor, radius)

    fig = plt.figure(constrained_layout=False, figsize=(12, 12))

    gs = fig.add_gridspec(3,2)
    time_ax = fig.add_subplot(gs[0, :])
    time_ax.set_xlabel('time [hours]')
    time_ax.set_ylabel('DDTEC [mTECU]')

    time_plot = time_ax.plot( np.arange(40),  np.arange(40), c='black')[0]

    dir_ax = fig.add_subplot(gs[1:, :], projection=wcs)
    dir_ax.coords[0].set_axislabel('Right Ascension (J2000)')
    dir_ax.coords[1].set_axislabel('Declination (J2000)')
    dir_ax.coords.grid(True, color='grey', ls='solid')
    polygons = []
    cmap = plt.cm.get_cmap('PuOr')
    norm = plt.Normalize(-1., 1.)
    colors = np.zeros(Nd)
    # colorize
    for color, region in zip(colors, regions):
        if np.size(color) == 1:
            if norm is None:
                color = cmap(color)
            else:
                color = cmap(norm(color))
        polygon = vertices[region]
        polygons.append(dir_ax.fill(*zip(*polygon), color=color, alpha=1., linewidth=4, edgecolor='black')[0])

    dir_ax.scatter(ref_dir[:, 0], ref_dir[:, 1], marker='*', color='black', zorder=19)

    # plt.plot(points[:,0], points[:,1], 'ko')
    dir_ax.set_xlim(vor.min_bound[0] - 0.1*radius, vor.max_bound[0] + 0.1*radius)
    dir_ax.set_ylim(vor.min_bound[1] - 0.1*radius, vor.max_bound[1] + 0.1*radius)

    def onkeyrelease(event):
        print('Pressed {} ({}, {})'.format(event.key, event.xdata, event.ydata))
        if event.key == 'n':
            print("Saving... going to next.")
            np.save(save_file, human_flags)
            next_loc = min(loc[0]+1, len(order))
            load_data(next_loc)
        if event.key == 'b':
            print("Saving... going to back.")
            np.save(save_file, human_flags)
            next_loc = max(loc[0]-1, 0)
            load_data(next_loc)
        if event.key == 's':
            print("Saving...")
            np.save(save_file, human_flags)

    def onclick(event):
        _, a, t = loc
        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              ('double' if event.dblclick else 'single', event.button,
               event.x, event.y, event.xdata, event.ydata))
        for i,region in enumerate(regions):
            if not human_flags[i,a,t]:
                polygons[i].set_zorder(10)
            if in_hull(vertices[region], np.array([event.xdata, event.ydata])):
                print("In region {}".format(i))
                if event.button == 1:
                    if human_flags[i,a,t] == -1 or human_flags[i,a,t] == 0:
                        human_flags[i, a, t] = 1
                    if human_flags[i, a, t] == 1:
                        human_flags[i, a, t] = 0
                    if human_flags[i,a,t] == 1:
                        polygons[i].set_edgecolor('red')
                    else:
                        polygons[i].set_edgecolor('green')
                    polygons[i].set_zorder(11)
                if event.button == 3:
                    start = max(0, t - 20)
                    stop = min(Nt, t + 20)
                    time_plot.set_data(times[start:stop], tec[0,i,a,start:stop])
                    time_ax.set_xlim(times[start:stop].min(), times[start:stop].max())
                    time_ax.set_ylim(tec[0,i,a,start:stop].min()-5., tec[0,i,a,start:stop].max()+5.)
                fig.canvas.draw()
                # break

    cid_click = fig.canvas.mpl_connect('button_press_event', onclick)
    cid_key = fig.canvas.mpl_connect('key_release_event', onkeyrelease)

    #Na, Nt

    search_first = np.where(np.any(guess_flags, axis=0))
    search_second = np.where(np.any(np.logical_not(guess_flags), axis=0))
    search = [list(sf)+list(ss) for sf, ss in zip(search_first, search_second)]
    order = list(np.random.choice(len(search_first[0]), len(search_first[0]), replace=False)) + \
            list(len(search_first[0])+np.random.choice(len(search_second[0]), len(search_second[0]), replace=False))
    loc = [0,0,0]

    def load_data(next_loc):
        loc[0] = next_loc
        o = order[next_loc]
        a = search[0][o]
        t = search[1][o]
        loc[1] = a
        loc[2] = t
        print("Looking a ant{:02d} and time {}".format(a, t))
        vmin, vmax = np.min(tec[0, :, a, t]), np.max(tec[0, :, a, t])
        vmin, vmax = min(vmin, -vmax), max(vmax, -vmin)
        norm = plt.Normalize(vmin, vmax)
        for i, p in enumerate(polygons):
            p.set_facecolor(cmap(norm(tec[0, i, a, t])))
            if guess_flags[i, a, t]:
                p.set_edgecolor('cyan')
                p.set_zorder(11)
            else:
                p.set_edgecolor('black')
                p.set_zorder(10)
        fig.canvas.draw()

    load_data(0)
    plt.show()

if __name__ == '__main__':
    dp = '/net/nederrijn/data1/albert/screens/root/L562061/download_archive/L562061_DDS4_full_merged.h5'
    ref_img = '/net/nederrijn/data1/albert/screens/root/L562061/download_archive/image_full_ampphase_di_m.NS.mask01.fits'
    click_through(dp, ref_img)