import rawpy
from astropy.io import fits
import numpy as np

# Change filename and/or path here
filename = "NEFs/005_2022-05-29_013427_A_DSC_0934.NEF"
filename_clean = filename[:-3]

raw = rawpy.imread(filename)
rgb = raw.postprocess()
r = rgb[:,:,0]
g = rgb[:,:,1]
b = rgb[:,:,2]

r = fits.PrimaryHDU(data=r).writeto(filename_clean+'_red.fits')
g = fits.PrimaryHDU(data=g).writeto(filename_clean+'_green.fits')
b = fits.PrimaryHDU(data=b).writeto(filename_clean+'_blue.fits')
