#!/usr/bin/env python
import casacore.tables as pt
import subprocess
import os, sys
import numpy as np
import argparse
import pyregion
from astropy.io import fits
from astropy.wcs import WCS
import glob
from DDFacet.Other import MyPickle
from DDFacet.ToolsDir.ModToolBox import EstimateNpix


def cmd_call(cmd):
    print("{}".format(cmd))
    exit_status = subprocess.call(cmd, shell=True)
    if exit_status:
        raise ValueError("Failed to  run: {}".format(cmd))

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





def flatten(f):
    """ Flatten a fits file so that it becomes a 2D image. Return new header and data """

    naxis = f[0].header['NAXIS']
    if naxis < 2:
        raise ValueError('Can\'t make map from this')
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
    parser.add_argument('--ncpu', help='number of cpu to use, default=34', default=32, type=int)
    parser.add_argument('--keeplongbaselines',
                        help='Use a Selection-UVRangeKm=[0.100000,5000.000000] instead of the DR2 default',
                        type='bool', default=False)
    parser.add_argument('--chunkhours', help='Data-ChunkHours for DDF.py (only used with --h5sols)',
                        default=8.5, type=float)
    parser.add_argument('--npix_out', help='Requested central box size to keep. Will be adjusted to work.',
                        default=10000, type=int)
    parser.add_argument('--working_dir', help='Where to perform the subtract.',
                        default=None, type=str, required=True)
    parser.add_argument('--data_dir', help='Where data is.',
                        default=None, type=str, required=True)
    parser.add_argument('--predict_column', help='Where the predicted data will go.',
                        default="PREDICT_SUB", type=str, required=False)
    parser.add_argument('--sub_column', help='Where the subtracted data will go.',
                        default="DATA_RESTRICTED", type=str, required=False)


def get_filenames(data_dir):
    print("Locating archive mask, dico, and clustercat.")
    fullmask = os.path.join(data_dir, os.path.basename('image_full_ampphase_di_m.NS.mask01.fits'))
    indico = os.path.join(data_dir, os.path.basename('image_full_ampphase_di_m.NS.DicoModel'))
    clustercat = os.path.join(data_dir, os.path.basename('image_dirin_SSD_m.npy.ClusterCat.npy'))
    mslist_file = os.path.join(data_dir, 'mslist.txt')
    if not os.path.isfile(fullmask):
        raise IOError("Missing mask {}".format(fullmask))
    if not os.path.isfile(indico):
        raise IOError("Missing dico model {}".format(indico))
    if not os.path.isfile(clustercat):
        raise IOError("Missing clustercat {}".format(clustercat))
    if not os.path.isfile(mslist_file):
        raise IOError("Missing mslist_file {}".format(mslist_file))
    mslist = []
    print('Reading {}'.format(mslist_file))
    with open(mslist_file, 'r') as f:
        for line in f.readlines():
            mslist.append(line.strip())
    return mslist_file, mslist, fullmask, indico, clustercat

def fix_dico_shape(fulldico, outdico, NPixOut):

    # dico_model = 'image_full_ampphase_di_m.NS.DicoModel'
    # save_dico = dico_model.replace('.DicoModel', '.restricted.DicoModel')
    # NPixOut = 10000

    dico = MyPickle.Load(fulldico)

    NPix = dico['ModelShape'][-1]
    NPix0, _ = EstimateNpix(float(NPix), Padding=1)
    if NPix != NPix0:
        raise ValueError("NPix != NPix0")
    print("Changing image size: %i -> %i pixels" % (NPix, NPixOut))
    xc0 = NPix // 2
    xc1 = NPixOut // 2
    dx = xc0 - xc1
    DCompOut = {}
    for k, v in dico.items():
        if k == 'Comp':
            DCompOut['Comp'] = {}
        DCompOut[k] = v
    DCompOut["Type"] = "SSD"

    N, M, _, _ = dico['ModelShape']
    DCompOut['ModelShape'] = [N, M, NPixOut, NPixOut]
    for (x0, y0) in dico['Comp'].keys():
        x1 = x0 - dx
        y1 = y0 - dx
        c0 = (x1 >= 0) & (x1 < NPixOut)
        c1 = (y1 >= 0) & (y1 < NPixOut)
        if c0 & c1:
            print("Mapping (%i,%i)->(%i,%i)" % (x0, y0, x1, y1))
            DCompOut['Comp'][(x1, y1)] = dico['Comp'][(x0, y0)]
    print("Saving in {}".format(outdico))
    MyPickle.Save(DCompOut, outdico)

def make_filtered_dico(region_mask, full_dico_model, masked_dico_model,npix_out=10000):
    """
    Filter dico model to only include sources in mask.

    :param region_mask:
    :param full_dico_model:
    :param masked_dico_model:
    :return:
    """
    print("Making dico containing only calibrators: {}".format(masked_dico_model))
    cmd = 'MaskDicoModel.py --MaskName={region_mask} --InDicoModel={full_dico_model} --OutDicoModel={masked_dico_model} --InvertMask=1'.format(
        region_mask=region_mask, full_dico_model=full_dico_model, masked_dico_model=masked_dico_model)
    cmd_call(cmd)
    if not os.path.isfile(masked_dico_model):
        raise IOError("Failed to make {}".format(masked_dico_model))
    fix_dico_shape(masked_dico_model, masked_dico_model, npix_out)

def make_predict_mask(infilename, ds9region, outfilename,npix_out=10000):
    """
    Make mask that is `infilename` everywhere except in regions specified by `ds9region` which is zero.
    :param infilename:
    :param ds9region:
    :param outfilename:
    :return:
    """
    print('Making: {}'.format(outfilename))
    npix_in = getimsize(infilename)
    s="image;box({},{},{},{},0)".format(npix_in//2, npix_in//2,npix_out, npix_out)
    with open(ds9region,'w') as f:
        f.write(s)
    hdu = fits.open(infilename)
    hduflat = flatten(hdu)
    r = pyregion.parse(s)
    manualmask = r.get_mask(hdu=hduflat)
    hdu[0].data[0][0][np.where(manualmask == True)] = 0.0
    hdu.writeto(outfilename, overwrite=True)
    if not os.path.isfile(outfilename):
        raise IOError("Did not successfully create {}".format(outfilename))

def make_predict_dico(indico, predict_dico, predict_mask,npix_out=10000):
    print("Making dico containing all but calibrators: {}".format(predict_dico))
    cmd_call(
        "MaskDicoModel.py --MaskName={} --InDicoModel={} --OutDicoModel={}".format(predict_mask, indico, predict_dico))
    if not os.path.isfile(predict_dico):
        raise IOError("Failed to make {}".format(predict_dico))


def cleanup_working_dir(working_dir):
    print("Deleting cache since we're done.")
    for f in glob.glob(os.path.join(working_dir,"*.ddfcache")):
        cmd_call("rm -r {}".format(f))

def main(data_dir, working_dir, ncpu, keeplongbaselines, chunkhours, predict_column, sub_column, npix_out):
    data_dir = os.path.abspath(data_dir)
    working_dir = os.path.abspath(working_dir)
    try:
        os.makedirs(working_dir)
    except:
        pass
    try:
        os.makedirs(os.path.join(working_dir, 'SOLSDIR'))
    except:
        pass
    os.chdir(working_dir)
    solsdir = os.path.join(data_dir, 'SOLSDIR')
    mslist_file, mslist, fullmask, indico, clustercat = get_filenames(data_dir)
    predict_dico = os.path.join(data_dir, 'image_full_ampphase_di_m.NS.not_{sub_column}.DicoModel'.format(sub_column=sub_column))
    filtered_dico = os.path.join(data_dir, 'image_full_ampphase_di_m.NS.{sub_column}.DicoModel'.format(sub_column=sub_column))
    predict_mask = os.path.join(data_dir, 'predict_mask_{sub_column}.fits'.format(sub_column=sub_column))  # just a name, can be anything
    region_file = os.path.join(working_dir,'box_subtract_region.reg')
    if keeplongbaselines:
        uvsel = "[0.100000,5000.000000]"
    else:
        uvsel = "[0.100000,1000.000000]"
    robust = -0.5
    imagecell = 1.5
    data_colname = 'DATA'
    columnchecker(mslist, data_colname)
    imagenpix = getimsize(fullmask)
    npix_out, _ = EstimateNpix(float(npix_out), Padding=1)
    # predict
    if os.path.isfile(predict_dico):
        os.unlink(predict_dico)
    if os.path.isfile(predict_mask):
        os.unlink(predict_mask)
    if os.path.isfile(filtered_dico):
        os.unlink(filtered_dico)
    make_predict_mask(fullmask, region_file, predict_mask, npix_out=npix_out)
    make_predict_dico(indico, predict_dico, predict_mask, npix_out=npix_out)
    make_filtered_dico(predict_mask, indico, filtered_dico, npix_out=npix_out)

    args = dict(chunkhours=chunkhours, mslist_file=mslist_file, data_colname=data_colname, ncpu=ncpu,
                clustercat=clustercat,
                robust=robust, imagenpix=imagenpix, imagecell=imagecell, predict_mask=predict_mask, predict_dico=predict_dico, uvsel=uvsel,
                solsdir=solsdir, predict_column=predict_column)
    print("Predicting...")
    cmd_call("DDF.py --Output-Name=image_full_ampphase_di_m.NS_SUB --Data-ChunkHours={chunkhours} --Data-MS={mslist_file} \
    --Deconv-PeakFactor=0.001000 --Data-ColName={data_colname} --Parallel-NCPU={ncpu} --Facets-CatNodes={clustercat} \
    --Beam-CenterNorm=1 --Deconv-Mode=SSD --Beam-Model=LOFAR --Beam-LOFARBeamMode=A --Weight-Robust={robust} \
    --Image-NPix={imagenpix} --CF-wmax=50000 --CF-Nw=100 --Output-Also=onNeds --Image-Cell={imagecell} \
    --Facets-NFacets=11 --SSDClean-NEnlargeData=0 --Freq-NDegridBand=1 --Beam-NBand=1 --Facets-DiamMax=1.5 \
    --Facets-DiamMin=0.1 --Deconv-RMSFactor=3.000000 --SSDClean-ConvFFTSwitch 10000 --Data-Sort=1 --Cache-Dir=. \
    --Log-Memory=1 --Cache-Weight=reset --Output-Mode=Predict --Output-RestoringBeam=6.000000 --Freq-NBand=2 \
    --RIME-DecorrMode=FT --SSDClean-SSDSolvePars=[S,Alpha] --SSDClean-BICFactor=0 --Mask-Auto=1 --Mask-SigTh=5.00 \
    --Mask-External={predict_mask} --DDESolutions-GlobalNorm=None --DDESolutions-DDModeGrid=AP \
    --DDESolutions-DDModeDeGrid=AP --DDESolutions-DDSols=[DDS3_full_smoothed,DDS3_full_slow] \
    --Predict-InitDicoModel={predict_dico} --Selection-UVRangeKm={uvsel} --GAClean-MinSizeInit=10 --Cache-Reset=1 \
    --Beam-Smooth=1 --Predict-ColName={predict_column} --DDESolutions-SolsDir={solsdir}".format(**args))
    # subtract
    print("Subtracting...")
    for ms in mslist:
        with pt.table(ms, readonly=False, ack=True) as t:
            colnames = t.colnames()
            if sub_column not in colnames:
                # Append new column containing all sources
                desc = t.getcoldesc(data_colname)
                newdesc = pt.makecoldesc(sub_column, desc)
                newdmi = t.getdminfo(data_colname)
                newdmi['NAME'] = 'Dysco' + sub_column
                t.addcols(newdesc, newdmi)

            for row in range(0, t.nrows(), 3000000):
                print('Reading {}'.format(predict_column))
                f = t.getcol(predict_column, startrow=row, nrow=3000000, rowincr=1)
                print('Reading', data_colname)
                d = t.getcol(data_colname, startrow=row, nrow=3000000, rowincr=1)

                print('Writing', sub_column)
                t.putcol(sub_column, d - f, startrow=row, nrow=3000000, rowincr=1)

    addextraweights(mslist)

    cleanup_working_dir(working_dir)

def test_main():
    main(data_dir='/home/albert/nederrijn_1/screens/root/L562061/download_archive',
         working_dir='/home/albert/nederrijn_1/screens/root/L562061/subtract_outside_pb',
         ncpu=56,
         keeplongbaselines=False,
         chunkhours=8.5,
         predict_column='PREDICT_SUB',
         sub_column='DATA_RESTRICTED',
         npix_out=10000
         )


if __name__ == '__main__':
    if len(sys.argv) == 1:
        test_main()
        exit(0)
    parser = argparse.ArgumentParser(
        description='Keep soures inside box region, subtract everything else and create new ms',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_args(parser)
    flags, unparsed = parser.parse_known_args()
    print("Running with:")
    for option, value in vars(flags).items():
        print("    {} -> {}".format(option, value))
    main(**vars(flags))
