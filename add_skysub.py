import sys
import os
import glob
import numpy as np
from astropy.io import fits

# this function will create a linearely binned version
# of a particular spectrum
from srebin import linlin

"""Add rebinned sky_subtracted to the x_multi*.fits"""

ff = glob.glob( "../xsky/2019????v???/exp0?/x_*.fits")

pattern = "/work/03946/hetdex/maverick/red1/reductions/{}/virus/virus0000{}/{}/virus/{}"

def get_rebinned(fin ,extensions=['spectrum', 'sky_spectrum', 'fiber_to_fiber'], start = 3494.74, step =  1.9858398, stop = 5500.):

	"""    #print("Reading {}".format(fin))
    hdu = fits.open(fin)

    wl = hdu['wavelength'].data

    #start,stop = 3503.9716796, 5396.477
    N = int( np.ceil( (stop - start)/step ) )

    rebinned = {}
	"""

	#start,stop = 3503.9716796, 5396.477
	N = int( np.ceil( (stop - start)/step ) )

	#print("Reading {}".format(fin))
	hdu = fits.open(fin)

	wl = hdu['wavelength'].data

	#start,stop = 3503.9716796, 5396.477
	rebinned = {}
	for ext in extensions:
		#for j in range(hdu[ext].data.shape[1]): # This might cause big errors...
		#isnull = np.unique(hdu[ext].data[:,j])
		#isnull = isnull[np.where(isnull != 0)]
		#if len(isnull)==0:
		#    hdu[ext].data[:, j] = np.ones(hdu[ext].data.shape[0])*np.nan
		#print("Rebinning {}".format(ext))

		new = np.zeros([wl.shape[0], N])
		hduextdata = hdu[ext].data
		for i in range(wl.shape[0]):
			w = wl[i,:]
			f = hduextdata[i,:]
			start = start
			step =  step
			stop = stop
			lw, lf = linlin(w, f, start, step, stop, dowarn=False)
			#lw = np.arange(start, stop, step)
			#lf = model_resampled_10A = spectres(lw, w, f)

			# hack as they may not all necessareyly have the same length
			new[i,:min(N, len(lf))] = lf[:min(N, len(lf))]

		rebinned[ext] = new
	return lw, rebinned

for fin in ff[:3]:

	night = fin.split("/")[2][:-4]
	shot = fin.split("/")[2][-3:]
	exp = fin.split("/")[3]
	name = fin.split("/")[-1][2:]

	newfin = pattern.format(night, shot, exp, name)
	
	hdu = fits.open(fin)
	try:
		skysub = hdu["sky_subtracted_rb"].data
		print("Found sky subtracted rb in "+fin)
		hdu.close()
	except KeyError:
		ww, rb = get_rebinned(newfin, extensions=["sky_subtracted"])
		hdu.append(fits.ImageHDU(rb["sky_subtracted"], name="sky_subtracted_rb"))
		hdu.writeto(fin, overwrite=True)
		print("Wrote to "+fin)
