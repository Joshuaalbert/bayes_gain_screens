#!/usr/bin/env python
import casacore.tables as pt
import os, sys
import numpy as np
import argparse
import pyregion
from astropy.io import fits
from astropy.wcs import WCS
from astropy.io import ascii
import glob


def getimsize(image):
    imsizeddf = None
    hdul = fits.open(image)
    his = hdul[0].header['HISTORY']
    for line in his:
        if 'Image-NPix' in line:
            imsizeddf = line

    if imsizeddf == 'None':
        print('Could not determine the image size, should have been 20000(?) or 6000(?)')
        sys.exit()

    imsizeddf = np.int(imsizeddf.split('=')[1])

    hdul.close()
    return imsizeddf


def get_solutions_timerange(sols):
    t = np.load(sols)['BeamTimes']
    return np.min(t), np.max(t)


def fixsymlinks(archive_dir, working_dir, obs_num):
    # Code from Tim for fixing symbolic links for DDS3_
    # dds3smoothed = glob.glob('SOLSDIR/*/*killMS.DDS3_full_smoothed*npz')
    print("Fixing symbolic links")
    dds3 = glob.glob(os.path.join(archive_dir, 'SOLSDIR/L{obs_num}*.ms/killMS.DDS3_full.sols.npz'.format(obs_num=obs_num)))
    for f in dds3:
        ms = os.path.basename(os.path.dirname(f))
        to_folder = os.path.join(working_dir, 'SOLSDIR', ms)
        try:
            os.makedirs(to_folder)
        except:
            pass
        for g in glob.glob(os.path.join(os.path.dirname(f), '*')):
            src = os.path.abspath(g)
            dst = os.path.join(to_folder, os.path.basename(g))
            if os.path.islink(dst):
                os.unlink(dst)
            print("Linking {} -> {}".format(src,dst))
            os.symlink(src,dst)
        start_time, _ = get_solutions_timerange(f)
        start_time = os.path.basename(
            glob.glob(os.path.join(archive_dir, 'DDS3_full_{}*_smoothed.npz'.format(int(start_time))))[0]).split('_')[2]
        src = os.path.join(archive_dir, 'DDS3_full_{}_smoothed.npz'.format(start_time))
        dst = os.path.join(to_folder, 'killMS.DDS3_full_smoothed.sols.npz')
        if os.path.islink(dst):
            os.unlink(dst)
        print("Linking {} -> {}".format(src, dst))
        os.symlink(src, dst)

def columnchecker(mslist, colname):
    for ms in mslist:
        t = pt.table(ms, ack=False)
        if colname not in t.colnames():
            print(colname, ' not present in ', ms)
            sys.exit()
        t.close()


def addextraweights(msfiles):
    '''
    Adds the column WEIGHT_SPECTRUM_FROM_IMAGING_WEIGHT from IMAGING_WEIGHT from DR2
    Input msfiles (list of ms)
    '''

    for ms in msfiles:
        with pt.table(ms, readonly=False) as ts:
            colnames = ts.colnames()
            if 'WEIGHT_SPECTRUM_FROM_IMAGING_WEIGHT' not in colnames:
                desc = ts.getcoldesc('WEIGHT_SPECTRUM')
                desc['name'] = 'WEIGHT_SPECTRUM_FROM_IMAGING_WEIGHT'
                ts.addcols(desc)

        with pt.table(ms, readonly=False) as ts:
            if 'IMAGING_WEIGHT' in colnames:
                iw = ts.getcol('IMAGING_WEIGHT')
                ws_tmp = ts.getcol('WEIGHT_SPECTRUM_FROM_IMAGING_WEIGHT')
                n, nfreq, npol = np.shape(ws_tmp)
                for i in range(npol):
                    print('Copying over correlation ', i, ms)
                    ws_tmp[:, :, i] = iw
                    ts.putcol('WEIGHT_SPECTRUM_FROM_IMAGING_WEIGHT', ws_tmp)


def mask_region(infilename, ds9region, outfilename):
    hdu = fits.open(infilename)
    hduflat = flatten(hdu)
    r = pyregion.open(ds9region)
    manualmask = r.get_mask(hdu=hduflat)
    hdu[0].data[0][0][np.where(manualmask == True)] = 0.0
    hdu.writeto(outfilename, overwrite=True)


def flatten(f):
    """ Flatten a fits file so that it becomes a 2D image. Return new header and data """

    naxis = f[0].header['NAXIS']
    if naxis < 2:
        raise RadioError('Can\'t make map from this')
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


def add_args(parser):
    parser.register("type", "bool", lambda v: v.lower() == "true")

    parser.add_argument('--region_file', help='boxfile, required argument', required=True, type=str)
    parser.add_argument('--ncpu', help='number of cpu to use, default=34', default=32, type=int)
    parser.add_argument('--keeplongbaselines',
                        help='Use a Selection-UVRangeKm=[0.100000,5000.000000] instead of the DR2 default',
                        type='bool', default=False)
    parser.add_argument('--chunkhours', help='Data-ChunkHours for DDF.py (only used with --h5sols)',
                        default=8.5, type=float)
    parser.add_argument('--obs_num', help='Obs number L*',
                        default=None, type=int, required=True)

    parser.add_argument('--archive_dir', help='Where are the archives stored.',
                        default=None, type=str, required=True)
    parser.add_argument('--working_dir', help='Where to perform the subtract.',
                        default=None, type=str, required=True)


def copy_archives(archive_dir, working_dir, obs_num):
    print("Copying archives.")
    archive_fullmask = os.path.join(archive_dir, 'image_full_ampphase_di_m.NS.mask01.fits')
    archive_indico = os.path.join(archive_dir, 'image_full_ampphase_di_m.NS.DicoModel')
    archive_clustercat = os.path.join(archive_dir, 'image_dirin_SSD_m.npy.ClusterCat.npy')
    fullmask = os.path.join(working_dir, os.path.basename(archive_fullmask))
    indico = os.path.join(working_dir, os.path.basename(archive_indico))
    clustercat = os.path.join(working_dir, 'image_dirin_SSD_m.npy.ClusterCat.npy')
    os.system('rsync -auvP {} {}'.format(archive_fullmask, fullmask))
    os.system('rsync -auvP {} {}'.format(archive_indico, indico))
    os.system('rsync -auvP {} {}'.format(archive_clustercat, clustercat))
    mslist = sorted(glob.glob(os.path.join(archive_dir, 'L{obs_num}*_SB*.ms.archive'.format(obs_num=obs_num))))
    print('Found archives files:\n{}'.format(mslist))
    outms = []
    for ms in mslist:
        outname = os.path.join(working_dir, os.path.basename(ms.rstrip('.archive')))
        cmd = 'rsync -auvP --delete {}/ {}/'.format(ms, outname)
        print(cmd)
        os.system(cmd)
        outms.append(outname)
    mslist_file = os.path.join(working_dir, 'mslist.txt')
    with open(mslist_file, 'w') as f:
        for ms in outms:
            f.write('{}\n'.format(ms))
    return mslist_file, outms, fullmask, indico, clustercat


def main(archive_dir, working_dir, obs_num, region_file, ncpu, keeplongbaselines, chunkhours):
    archive_dir = os.path.abspath(archive_dir)
    working_dir = os.path.abspath(working_dir)
    region_file = os.path.abspath(region_file)
    try:
        os.makedirs(working_dir)
    except:
        pass
    try:
        os.makedirs(os.path.join(working_dir, 'SOLSDIR'))
    except:
        pass
    os.chdir(working_dir)
    solsdir = os.path.join(working_dir, 'SOLSDIR')
    mslist_file, mslist, fullmask, indico, clustercat = copy_archives(archive_dir, working_dir, obs_num)
    outdico = os.path.join(working_dir, 'image_full_ampphase_di_m_SUB.NS.DicoModel')
    outmask = os.path.join(working_dir, 'cutoutmask.fits')  # just a name, can be anything
    if not os.path.isfile(fullmask):
        raise IOError("Missing mask {}".format(fullmask))
    if not os.path.isfile(indico):
        raise IOError("Missing dico model {}".format(indico))
    if not os.path.isfile(clustercat):
        raise IOError("Missing clustercat {}".format(clustercat))
    fixsymlinks(archive_dir, working_dir, obs_num)
    if keeplongbaselines:
        uvsel = "[0.100000,5000.000000]"
    else:
        uvsel = "[0.100000,1000.000000]"
    robust = -0.5
    imagecell = 1.5
    data_colname = 'DATA'
    outcolname = 'DATA_SUB'
    columnchecker(mslist, data_colname)
    imagenpix = getimsize(fullmask)
    # predict
    if os.path.isfile(outdico):
        os.unlink(outdico)
    if os.path.isfile(outmask):
        os.unlink(outmask)
    print("Masking region with {}.".format(region_file))
    mask_region(fullmask, region_file, outmask)
    print("Masking dico model.")
    os.system("MaskDicoModel.py --MaskName={} --InDicoModel={} --OutDicoModel={}".format(outmask, indico, outdico))
    args = dict(chunkhours=chunkhours, mslist_file=mslist_file, data_colname=data_colname, ncpu=ncpu,
                clustercat=clustercat,
                robust=robust, imagenpix=imagenpix, imagecell=imagecell, outmask=outmask, outdico=outdico, uvsel=uvsel,
                solsdir=solsdir)
    print("Predicting...")
    os.system("DDF.py --Output-Name=image_full_ampphase_di_m.NS_SUB --Data-ChunkHours={chunkhours} --Data-MS={mslist_file} \
    --Deconv-PeakFactor=0.001000 --Data-ColName={data_colname} --Parallel-NCPU={ncpu} --Facets-CatNodes={clustercat} \
    --Beam-CenterNorm=1 --Deconv-Mode=SSD --Beam-Model=LOFAR --Beam-LOFARBeamMode=A --Weight-Robust={robust} \
    --Image-NPix={imagenpix} --CF-wmax=50000 --CF-Nw=100 --Output-Also=onNeds --Image-Cell={imagecell} \
    --Facets-NFacets=11 --SSDClean-NEnlargeData=0 --Freq-NDegridBand=1 --Beam-NBand=1 --Facets-DiamMax=1.5 \
    --Facets-DiamMin=0.1 --Deconv-RMSFactor=3.000000 --SSDClean-ConvFFTSwitch 10000 --Data-Sort=1 --Cache-Dir=. \
    --Log-Memory=1 --Cache-Weight=reset --Output-Mode=Predict --Output-RestoringBeam=6.000000 --Freq-NBand=2 \
    --RIME-DecorrMode=FT --SSDClean-SSDSolvePars=[S,Alpha] --SSDClean-BICFactor=0 --Mask-Auto=1 --Mask-SigTh=5.00 \
    --Mask-External={outmask} --DDESolutions-GlobalNorm=None --DDESolutions-DDModeGrid=AP \
    --DDESolutions-DDModeDeGrid=AP --DDESolutions-DDSols=[DDS3_full_smoothed,DDS3_full_slow] \
    --Predict-InitDicoModel={outdico} --Selection-UVRangeKm={uvsel} --GAClean-MinSizeInit=10 --Cache-Reset=1 \
    --Beam-Smooth=1 --Predict-ColName=PREDICT_SUB --DDESolutions-SolsDir={solsdir}".format(**args))
    # subtract
    print("Subtracting...")
    for ms in mslist:
        with pt.table(ms, readonly=False, ack=True) as t:
            colnames = t.colnames()
            if outcolname not in colnames:
                # Append new column containing all sources
                desc = t.getcoldesc(data_colname)
                newdesc = pt.makecoldesc(outcolname, desc)
                newdmi = t.getdminfo(data_colname)
                newdmi['NAME'] = 'Dysco' + outcolname
                t.addcols(newdesc, newdmi)

            for row in range(0, t.nrows(), 3000000):
                print('Reading PREDICT_SUB')
                f = t.getcol('PREDICT_SUB', startrow=row, nrow=3000000, rowincr=1)
                print('Reading', data_colname)
                d = t.getcol(data_colname, startrow=row, nrow=3000000, rowincr=1)

                print('Writing', outcolname)
                t.putcol(outcolname, d - f, startrow=row, nrow=3000000, rowincr=1)

    addextraweights(mslist)


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
