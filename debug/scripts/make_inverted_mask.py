from astropy.io import fits

def main(in_mask, out_mask):
    with fits.open(in_mask) as hdu:
        hdu[0].data = 1. - hdu[0].data
        hdu.writeto(out_mask)

if __name__ == '__main__':
    main('/home/albert/store/lockman/data/cutoutmask.fits', '/home/albert/store/lockman/data/cutoutmask_inverted.fits')
